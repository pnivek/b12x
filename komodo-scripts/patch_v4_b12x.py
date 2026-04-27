#!/usr/bin/env python3
"""pre_build script for DeepSeek-V4-Flash with b12x sparse MLA acceleration.

Derived from patch_v4_triton.py with these additional patches:
  - Swap vllm source repo: vllm-project/vllm -> pnivek/vllm-ds4-sm12x
    (set branch via VLLM_REF build_arg = ds4-sm120-b12x)
  - Drop VLLM_PRS=40899 (our fork already includes that work)
  - Append `pip install b12x @ git+https://github.com/pnivek/b12x.git@master`
    in the runner stage so import b12x works at runtime
"""

import datetime as dt
import os
import subprocess
import sys


print("=== pre_build: patch_v4_b12x.py ===")
REPO_DIR = os.getcwd()

# 1. Reset Dockerfile working tree
subprocess.run(["git", "reset", "--hard", "HEAD"], check=False,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.run(["git", "clean", "-fdx"], check=False,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

df_path = os.path.join(REPO_DIR, "Dockerfile")
with open(df_path) as f:
    content = f.read()
print(f"Dockerfile: {len(content.splitlines())} lines")

# 2. Cubins-cache fix for CUDA 13.x
if "id=cubins-cache," in content and "cubins-cache-v2," not in content:
    content = content.replace("id=cubins-cache,", "id=cubins-cache-v2,")
    print("  Fixed cubins-cache -> cubins-cache-v2")
elif "cubins-cache-v2," in content:
    print("  cubins-cache-v2 already set")

# 2b. Use a separate repo-cache id so we don't inherit the triton build's
#     vllm-project/vllm remote (which would refuse to find our fork branch).
#     v3 suffix: previous v1/v2 caches got left in a partial-submodule state
#     by cancelled build attempts. Bumping the id forces a fresh empty cache.
NEW_CACHE = "id=repo-cache-b12x-v4,"
if "id=repo-cache," in content and NEW_CACHE not in content:
    content = content.replace("id=repo-cache,", NEW_CACHE)
    print(f"  Forked repo-cache -> {NEW_CACHE.rstrip(',')} (fresh clone)")
elif NEW_CACHE in content:
    print(f"  {NEW_CACHE.rstrip(',')} already set")

# 3. Swap vLLM source repo URL to our fork
ORIG_VLLM_URL = "https://github.com/vllm-project/vllm.git"
FORK_VLLM_URL = "https://github.com/pnivek/vllm.git"
if ORIG_VLLM_URL in content:
    content = content.replace(ORIG_VLLM_URL, FORK_VLLM_URL)
    print(f"  Swapped vLLM source -> {FORK_VLLM_URL}")
elif FORK_VLLM_URL in content:
    print("  vLLM source already pointed at fork")
else:
    print("  WARNING: could not find vllm clone URL to patch", file=sys.stderr)

# 4. Replace runner-stage wheel bind-mount with COPY from builder stages
lines = content.splitlines(True)
in_runner = False
bind_mount_line_idx = None
bind_mount_removed = False
for i, line in enumerate(lines):
    if "AS runner" in line:
        in_runner = True
        print(f"  L{i+1}: Found runner stage")
        continue
    if not in_runner:
        continue
    if "--mount=type=bind,source=wheels,target=/workspace/wheels" in line:
        bind_mount_line_idx = i
        lines[i] = line.replace(
            "--mount=type=bind,source=wheels,target=/workspace/wheels",
            "",
        )
        stripped = lines[i].rstrip()
        if stripped.endswith("\\"):
            lines[i] = stripped + "\n"
        bind_mount_removed = True
        print(f"  L{i+1}: Removed wheels bind-mount from runner RUN")
        break

if bind_mount_removed:
    runner_start = next(i for i, l in enumerate(lines) if "AS runner" in l)
    replaced = 0
    for i in range(runner_start, len(lines)):
        if "uv pip install /workspace/wheels/*.whl" in lines[i]:
            lines[i] = lines[i].replace(
                "/workspace/wheels/*.whl",
                "/workspace/built-wheels/*.whl",
            )
            replaced += 1
    print(f"  Replaced {replaced} pip install path(s) -> built-wheels")

    if "COPY --from=vllm-builder /workspace/wheels /workspace/built-wheels/" not in "".join(lines):
        insert_idx = bind_mount_line_idx
        while insert_idx > 0 and not lines[insert_idx].lstrip().startswith("RUN"):
            insert_idx -= 1
        copy_lines = [
            "# Copy freshly-built wheels from builder stages\n",
            "COPY --from=vllm-builder /workspace/wheels /workspace/built-wheels/\n",
            "COPY --from=flashinfer-builder /workspace/wheels /workspace/built-wheels/\n",
            "\n",
        ]
        for j, cl in enumerate(copy_lines):
            lines.insert(insert_idx + j, cl)
        print(f"  Inserted COPY from builders at L{insert_idx+1}")
else:
    print("  WARNING: could not find runner wheels bind-mount", file=sys.stderr)

# 5. Append DeepGEMM (PR #318 + #319 SM120/121)
content = "".join(lines)
if "deepseek-ai/DeepGEMM" not in content:
    deepgemm_block = (
        "\n"
        "# Build DeepGEMM with SM120/SM121 support (PR #318 + #319)\n"
        "RUN cd /tmp && \\\n"
        "    rm -rf DeepGEMM && \\\n"
        "    git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git && \\\n"
        "    cd DeepGEMM && \\\n"
        "    git fetch origin pull/318/head:pr-318 && \\\n"
        "    git checkout 23ec7b5662101edbba60cf82a6661512a6759907 && \\\n"
        "    git fetch origin pull/319/head:pr-319 && \\\n"
        "    git checkout pr-319 -- csrc/apis/hyperconnection.hpp && \\\n"
        "    git checkout pr-319 -- csrc/indexing/main.cu && \\\n"
        "    git checkout pr-319 -- csrc/jit_kernels/heuristics/sm120.hpp && \\\n"
        "    git checkout pr-319 -- csrc/jit_kernels/impls/sm120_tf32_hc_prenorm_gemm.hpp && \\\n"
        "    git checkout pr-319 -- deep_gemm/include/deep_gemm/impls/sm120_tf32_hc_prenorm_gemm.cuh && \\\n"
        "    git checkout pr-319 -- deep_gemm/include/deep_gemm/mma/sm120.cuh && \\\n"
        "    pip install . --no-build-isolation\n"
    )
    lines.append(deepgemm_block)
    print("  Appended DeepGEMM build (PR #318 + #319)")
else:
    print("  DeepGEMM already present")

# 6. safetensors 0.8.0rc0 (F8_E8M0 dtype support)
content = "".join(lines)
if "safetensors==0.8.0rc0" not in content:
    lines.append("\n# Upgrade safetensors for F8_E8M0 dtype support\n")
    lines.append("RUN pip install --upgrade safetensors==0.8.0rc0\n")
    print("  Appended safetensors==0.8.0rc0 upgrade")
else:
    print("  safetensors already upgraded")

# 7. Install b12x from our public fork
content = "".join(lines)
B12X_PIP = "pip install --no-deps git+https://github.com/pnivek/b12x.git@master"
if "github.com/pnivek/b12x" not in content:
    lines.append("\n# Install b12x sparse MLA library (DSV4-Flash patches)\n")
    lines.append(f"RUN {B12X_PIP}\n")
    print("  Appended b12x install")
else:
    print("  b12x install already present")

# 8. Write patched Dockerfile
with open(df_path, "w") as f:
    f.writelines(lines)
print(f"  Dockerfile written ({len(lines)} lines)")

# 9. Build metadata
build_date = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
git_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
).decode().strip()
meta = (
    f"build_date: {build_date}\n"
    f"build_script: komodo-scripts/patch_v4_b12x.py\n"
    f"build_script_commit: {git_commit}\n"
    f"vllm_repo: pnivek/vllm-ds4-sm12x\n"
    f"vllm_ref: ds4-sm120-b12x\n"
    f"b12x_ref: pnivek/b12x@master\n"
    f"gpu_arch: 12.1a\n"
    f"base_image: nvidia/cuda:13.2.0-devel-ubuntu24.04\n"
    f"deepgemm: PR#318+PR#319\n"
    f"attention: triton-fallback + b12x (toggle via VLLM_B12X_MLA env)\n"
    f"safetensors: 0.8.0rc0\n"
)
with open(os.path.join(REPO_DIR, "build-metadata.yaml"), "w") as f:
    f.write(meta)
print(f"Build metadata written ({len(meta)} chars)")
print("=== patch_v4_b12x.py complete ===")
