#!/usr/bin/env python3
"""Create the head + worker stacks for vllm-deepseek-v4-b12x.

Clones the existing triton stack composes (so all the cluster-tested args
stay identical) and patches:
  - image      -> vllm-node-deepseek-v4-b12x:latest
  - service    -> vllm-deepseek-v4-{head|worker}-b12x
  - container  -> vllm_deepseek_v4_{head|worker}_b12x
  - environment + VLLM_B12X_MLA=1   (toggle b12x dispatch)

Idempotent: if a stack with the target name already exists, UpdateStack only.

Usage:
    KOMODO_API_KEY=... KOMODO_API_SECRET=... python3 create_b12x_stacks.py
"""

from __future__ import annotations

import json
import os
import sys

import requests


KOMODO_URL = os.environ.get("KOMODO_ADDRESS", "http://192.168.0.181:9120")
HEADERS = {
    "X-API-Key": os.environ["KOMODO_API_KEY"],
    "X-API-Secret": os.environ["KOMODO_API_SECRET"],
    "Content-Type": "application/json",
}

PAIRS = [
    {
        "src": "vllm-deepseek-v4-head-triton",
        "dst": "vllm-deepseek-v4-head-b12x",
        "src_service": "vllm-deepseek-v4-head-triton",
        "dst_service": "vllm-deepseek-v4-head-b12x",
        "src_container": "vllm_deepseek_v4_head_triton",
        "dst_container": "vllm_deepseek_v4_head_b12x",
    },
    {
        "src": "vllm-deepseek-v4-worker-triton",
        "dst": "vllm-deepseek-v4-worker-b12x",
        "src_service": "vllm-deepseek-v4-worker-triton",
        "dst_service": "vllm-deepseek-v4-worker-b12x",
        "src_container": "vllm_deepseek_v4_worker_triton",
        "dst_container": "vllm_deepseek_v4_worker_b12x",
    },
]


def call(endpoint: str, payload: dict) -> dict:
    resp = requests.post(f"{KOMODO_URL}{endpoint}", headers=HEADERS,
                         data=json.dumps(payload), timeout=30)
    if not resp.ok:
        sys.exit(f"{endpoint} failed: HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def find_stack(name: str) -> dict | None:
    for s in call("/read/ListStacks", {}):
        if s.get("name") == name:
            return s
    return None


def patch_compose(src_compose: str, p: dict) -> str:
    out = src_compose
    out = out.replace("vllm-node-deepseek-v4-triton:latest",
                      "vllm-node-deepseek-v4-b12x:latest")
    out = out.replace(p["src_service"], p["dst_service"])
    out = out.replace(p["src_container"], p["dst_container"])
    # Add VLLM_B12X_MLA=1 right after VLLM_TRITON_MLA_SPARSE=1 line
    if "VLLM_B12X_MLA" not in out:
        out = out.replace(
            "      - VLLM_TRITON_MLA_SPARSE=1",
            "      - VLLM_TRITON_MLA_SPARSE=1\n      - VLLM_B12X_MLA=1",
        )
    return out


def main() -> int:
    for p in PAIRS:
        src = call("/read/GetStack", {"stack": p["src"]})
        src_cfg = src.get("config", {})
        src_compose = src_cfg.get("file_contents", "")
        if not src_compose:
            sys.exit(f"source stack {p['src']} has empty file_contents")
        new_compose = patch_compose(src_compose, p)

        existing = find_stack(p["dst"])
        if existing:
            stack_id = existing["id"]
            print(f"Updating existing stack '{p['dst']}' (id={stack_id})")
        else:
            created = call("/write/CreateStack", {
                "name": p["dst"],
                "config": {
                    "server_id": src_cfg.get("server_id"),
                    "file_contents": new_compose,
                },
            })
            stack_id = created.get("_id", {}).get("$oid") or created.get("id")
            if not stack_id:
                sys.exit(f"CreateStack returned no id: {created}")
            print(f"Created stack '{p['dst']}' (id={stack_id})")

        # Always UpdateStack to lock in the full file_contents
        call("/write/UpdateStack", {
            "id": stack_id,
            "config": {
                "server_id": src_cfg.get("server_id"),
                "file_contents": new_compose,
                "project_name": src_cfg.get("project_name", "deepseek-v4"),
                "destroy_before_deploy": src_cfg.get("destroy_before_deploy", False),
                "auto_pull": True,
            },
        })
        print(f"  server_id: {src_cfg.get('server_id')}")
        print(f"  compose lines: {len(new_compose.splitlines())}")

    print("\nNext: trigger deploy via /execute/DeployStack once the image is built.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
