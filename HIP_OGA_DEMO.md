# FastFlowLM HIP OGA Demo

This branch provides an experimental GPU backend for FastFlowLM using ONNX
Runtime GenAI, HIP EP, and the `AMDGPU` execution provider.

## Source Base

The integration is based on FastFlowLM main commit:

```text
06513091bf84044a8f9d98ebccd86f04fe5b4ac2
```

The branch contains the recovered HIP OGA CLI/server integration, strict GPU
policy, external OGA model support, VLM request routing, tool-call response
handling, and the experimental text prefix cache.

## Build

Use an x64 Visual Studio developer prompt, CMake 3.22 or newer, the normal
FastFlowLM Windows dependencies, and an ABI-matched ONNX Runtime GenAI SDK.

```powershell
cmake -S src -B build -A x64 `
  -DFLM_VERSION=0.1.0 `
  -DNPU_VERSION=0.1.0 `
  -DFLM_ENABLE_HIP_OGA=ON `
  -DFLM_HIP_OGA_ONLY=OFF `
  -DFLM_USE_HRX=OFF `
  -DONNXRUNTIME_GENAI_HOME=C:\path\to\oga-sdk

cmake --build build --config Release --target flm -- /m:8
```

## Runtime

Use an OGA/ONNX model directory containing `genai_config.json`. Keep the
matching ONNX Runtime, ONNX Runtime GenAI, HIP EP, and HIP runtime DLLs beside
`flm.exe` so Windows does not load an incompatible system runtime first.

```powershell
$env:FLM_BACKEND = 'hip'
$env:FLM_HIP_OGA_OFFLOAD_POLICY = 'strict_gpu'
$env:FLM_HIP_OGA_DISABLE_CPU_FALLBACK = '1'
$env:ORT_DISABLE_CPU_EP_FALLBACK = '1'

.\flm.exe validate --backend hip --json
.\flm.exe run C:\path\to\oga-model `
  --backend hip --ctx-len 512 --prompt Hi --max-tokens 1
```

## Experimental Prefix Cache

The text prefix cache is off by default. Enable it only for an experimental
server run:

```powershell
$env:FLM_HIP_OGA_PREFIX_CACHE = 'experimental'
```

The cache verifies token prefixes, restores a prior generator checkpoint, and
appends only the uncached suffix. It invalidates on incompatible model,
context, sampling, search, or capacity changes. VLM/image/audio requests force
a cache miss.

Qualified two-turn evidence observed 24 cached prompt tokens on the second
experimental request, while the equivalent control request reported zero.

## Qualification Boundary

The experimental package was validated with:

- strict AMDGPU/HIP execution and CPU fallback disabled;
- control versus cache-hit semantic checks;
- divergence, no-suffix, sampling, and max-token invalidation cases;
- real-image VLM forced cache miss;
- two 50-cycle runs with second-turn cache hits.

Known limits:

- provider-fault fallback was not injected on hardware;
- one 50-cycle sample showed substantial working-set growth;
- no production, default-on, long-duration reliability, or performance claim;
- the qualified package context was 512 tokens.

Model weights and runtime packages are intentionally not stored in this source
branch. Confirm model and runtime redistribution terms before publishing binary
release assets.
