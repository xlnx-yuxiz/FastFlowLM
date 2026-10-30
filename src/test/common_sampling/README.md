# Common sampling tests

The default standalone suite checks shared numerical processing, request
normalization and the OGA single-candidate mapping. It needs a C++17 compiler
and CMake 3.22+, with no OGA/XRT runtime or model execution:

```shell
cmake -S src/test/common_sampling -B build-common-sampling
cmake --build build-common-sampling --config Release
ctest --test-dir build-common-sampling -C Release --output-on-failure
```

To add the actual native `Sampler`/BF16-buffer tests, configure a separate
directory with `FLM_TEST_NATIVE_SAMPLER=ON` and the installed development
dependencies:

```shell
cmake -S src/test/common_sampling -B build-native-sampling \
  -DFLM_TEST_NATIVE_SAMPLER=ON \
  -DXRT_INCLUDE_DIR=/path/to/xrt/include \
  -DXRT_LIB_DIR=/path/to/xrt/lib \
  -DBOOST_INCLUDE_DIR=/path/to/boost
cmake --build build-native-sampling --config Release
ctest --test-dir build-native-sampling -C Release --output-on-failure
```

On Windows, use a Visual Studio developer prompt and Windows dependency paths.
These five targets include both the common and legacy native sampler cases.
They use synthetic logits and do not establish real NPU model qualification.

The HTTP fixture uses an already running HIP server and Python's standard
library. For a compatible tool-capable Qwen2.5 Coder package, start the server
with `--sampler common`, then run:

```shell
python src/test/hip_oga_backend_static/test_common_sampling_api.py \
  --base-url http://127.0.0.1:8000 \
  --model /path/to/qualified-oga-model \
  --startup-sampler common --tool-routing template \
  --output /path/to/new-sampling-results
```

For the tested Gemma package's unsupported tool route, use
`--tool-routing unsupported`. Repeat with a freshly started
`--sampler backend` server and matching `--startup-sampler backend` to check
startup defaults. The caller owns server startup/shutdown, codec and cache
settings, runtime placement and log collection. The supplied startup label is
not independently attested by this fixture. A passing fixture alone does not
cover the full CLI/runtime/cancellation/resource qualification matrix.
