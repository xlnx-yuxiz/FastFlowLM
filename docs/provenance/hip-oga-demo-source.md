# HIP OGA Demo Source Provenance

## Lineage

| Stage | Commit |
| --- | --- |
| FastFlowLM main base | `06513091bf84044a8f9d98ebccd86f04fe5b4ac2` |
| Recovered HIP OGA demo integration | `b8401cc40ce317bef31b1640a996166b914b9c21` |
| Experimental PrefixCache | `a717ff581edaf3425bb589ec6e2642f8416d50ca` |
| PrefixCache capacity correction | `07fefdbc699eebff68672737e6bd52f7a6e50b83` |

The public demo branch squashes the integration changes onto the original main
base. Generated builds, model weights, runtime archives, machine-specific
paths, and internal test evidence are excluded.

## Package Receipt

The qualified internal experimental package had:

```text
Archive SHA-256:
753fbfd10cf7713b121479e337f6f1a5e445bea09e9f07e7f8e74606e567de95

Packaged flm.exe SHA-256:
0d4f617c1ccc59a615f61363dcecc71106eaa3f845bf361be8cd5f346610527e

Independent rebuild SHA-256:
d8251a21a6e07158efec734b3e93c423ecfedcedd6ea147423657e8bf07487cb
```

The independent rebuild was not bit-identical to the packaged executable, but
it passed the required strict HIP two-turn cache smoke. This branch publishes
source only; it does not publish the internal package or model weights.

## Runtime Contract

- ONNX Runtime product version: `1.30.0`
- ONNX Runtime GenAI DLL SHA-256:
  `873d84c377163007b65544344a34fc69b80e9466fdaa71f937877be55dc15503`
- Execution provider: `AMDGPU`
- Profile: `hip`
- CPU EP fallback: disabled for the strict qualification path
- Qualified context: 512 tokens

Use a matching runtime envelope. Loading a system ONNX Runtime built for an
older API version is not supported.
