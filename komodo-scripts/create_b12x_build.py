#!/usr/bin/env python3
"""Create the Komodo build `vllm-node-deepseek-v4-b12x` by cloning the
existing `vllm-node-deepseek-v4-triton` build's config and patching:

  - name -> vllm-node-deepseek-v4-b12x
  - pre_build command -> patch_v4_b12x.py
  - build_args: drop VLLM_PRS=40899 (our fork already includes that work),
    set VLLM_REF=ds4-sm120-b12x, fresh CACHEBUST values

Usage:
    KOMODO_API_KEY=... KOMODO_API_SECRET=... python3 create_b12x_build.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import requests


KOMODO_URL = os.environ.get("KOMODO_ADDRESS", "http://192.168.0.181:9120")
HEADERS = {
    "X-API-Key": os.environ["KOMODO_API_KEY"],
    "X-API-Secret": os.environ["KOMODO_API_SECRET"],
    "Content-Type": "application/json",
}

TRITON_BUILD_ID = "69edb20ac63eb33817ce9605"
NEW_BUILD_NAME = "vllm-node-deepseek-v4-b12x"


def call(endpoint: str, payload: dict) -> dict:
    url = f"{KOMODO_URL}{endpoint}"
    resp = requests.post(url, headers=HEADERS, data=json.dumps(payload), timeout=30)
    if not resp.ok:
        sys.exit(f"{endpoint} failed: HTTP {resp.status_code}: {resp.text[:300]}")
    try:
        return resp.json()
    except json.JSONDecodeError:
        sys.exit(f"{endpoint} returned non-JSON: {resp.text[:300]}")


def find_build(name: str) -> dict | None:
    builds = call("/read/ListBuilds", {})
    for b in builds:
        if b.get("name") == name:
            return b
    return None


def main() -> int:
    triton = call("/read/GetBuild", {"build": TRITON_BUILD_ID})
    triton_cfg = triton.get("config", {})
    print(f"Cloning triton config: {len(triton_cfg)} fields")

    existing = find_build(NEW_BUILD_NAME)
    if existing:
        build_id = existing["id"]
        print(f"Build '{NEW_BUILD_NAME}' already exists (id={build_id}); will UpdateBuild only.")
    else:
        created = call("/write/CreateBuild", {
            "name": NEW_BUILD_NAME,
            "config": {"builder_id": triton_cfg.get("builder_id")},
        })
        build_id = created.get("_id", {}).get("$oid") or created.get("id")
        if not build_id:
            sys.exit(f"CreateBuild returned no id: {created}")
        print(f"Created build '{NEW_BUILD_NAME}' id={build_id}")

    cachebust = str(int(time.time()))
    new_cfg = dict(triton_cfg)
    new_cfg.update({
        "pre_build": {
            "path": "",
            "command": "python3 /home/pnivek/komodo-scripts/patch_v4_b12x.py\n",
        },
        "build_args": (
            f'CACHEBUST_FLASHINFER = "{cachebust}"\n'
            f'CACHEBUST_VLLM = "{cachebust}"\n'
            f'VLLM_REF = "ds4-sm120-b12x"\n'
        ),
    })
    # CreateBuild may accept builder_id but UpdateBuild needs the full config.
    updated = call("/write/UpdateBuild", {
        "id": build_id,
        "config": new_cfg,
    })
    print(f"Updated build '{NEW_BUILD_NAME}'")
    print(f"  pre_build: {new_cfg['pre_build']['command'].strip()}")
    print(f"  build_args:\n    " + "\n    ".join(new_cfg['build_args'].strip().split("\n")))
    print(f"\nTo trigger:")
    print(f"  curl -X POST {KOMODO_URL}/execute/RunBuild \\")
    print(f"    -H 'X-API-Key: $KOMODO_API_KEY' -H 'X-API-Secret: $KOMODO_API_SECRET' \\")
    print(f"    -H 'Content-Type: application/json' -d '{{\"build\":\"{build_id}\"}}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
