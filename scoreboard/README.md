# ONNX Backend Scoreboard Integration

Tracks progress on adding burn-onnx to the
[ONNX Backend Scoreboard](https://onnx.ai/backend-scoreboard/).

See: [tracel-ai/burn-onnx#299](https://github.com/tracel-ai/burn-onnx/issues/299)

## Architecture

burn-onnx is a code generator, not a runtime. The ONNX scoreboard requires a Python
`onnx.backend.base.Backend` interface. Our approach (modeled after
[emx-onnx-cgen](https://github.com/emmtrix/emx-onnx-cgen/) and
[tract](https://github.com/sonos/tract)):

```
Per test case:

  Python backend.prepare(model)
    |
    v
  Save model.onnx -> onnx2burn codegen -> model.rs + model.bpk
    |
    v
  Parse forward() signature from model.rs
    |
    v
  Generate main.rs harness (type-safe Rust I/O for this model's signature)
    |
    v
  cargo build (debug, incremental; burn deps pre-compiled)
    |
    v
  Return BurnOnnxBackendRep(compiled_binary)

  Python backend.run(inputs)
    |
    v
  Serialize numpy inputs -> binary files + manifest.json
    |
    v
  Execute compiled binary (reads manifest, runs model, writes outputs)
    |
    v
  Deserialize output binary files -> numpy arrays
```

The Docker image pre-compiles burn and all Rust dependencies once during `docker build`. Per-model
incremental builds only recompile the generated model code.

## Files

| File            | Purpose                                       | Destination in scoreboard repo         |
| --------------- | --------------------------------------------- | -------------------------------------- |
| `backend.py`    | Python ONNX Backend wrapper                   | `backends/burn-onnx/backend.py`        |
| `runner/`       | Rust template project (pre-compiled deps)     | `backends/burn-onnx/runner/`           |
| `Dockerfile`    | Docker build for CI                           | `runtimes/burn-onnx/stable/Dockerfile` |
| `config.json`   | Config entry (merge into `setup/config.json`) | `setup/config.json`                    |
| `test_local.py` | Local test runner (not submitted upstream)    | N/A                                    |

## Local Testing

```bash
# 1. Build onnx2burn CLI
cargo build --release -p burn-onnx --bin onnx2burn
export ONNX2BURN="$(pwd)/target/release/onnx2burn"

# 2. Pre-compile runner in debug mode (compiles burn deps, takes a few minutes
# first time). Debug profile matches what backend.py rebuilds per model, so the
# cached dep artifacts get reused on every subsequent test case.
cd scoreboard/runner
cargo build
cd ..

# 3. Install Python deps
uv pip install onnx pytest numpy

# 4. Run tests
cd scoreboard
uv run python test_local.py -k "test_abs" -v          # single op
uv run python test_local.py -k "not _cuda" -v          # all CPU tests
uv run python test_local.py --collect-only              # list tests
```

## Preliminary Results

From a 200-test evenly-sampled run across all 1599 CPU tests:

- 180/200 passed (90%)
- ~1 second per test (incremental compile + inference)
- Estimated full suite: ~27 minutes, well within 90-minute scoreboard timeout
- Failures are in burn-onnx op correctness (reduce ops, resize, tril), not integration issues

## Known Blockers / Considerations

1. **Compile time per model**: Each ONNX test model requires an incremental Rust build. With
   pre-compiled deps, this takes ~1 second per model. Full suite of ~1600 tests takes ~27 minutes,
   within the 90-minute scoreboard timeout.

2. **Unsupported ops**: Any ONNX op not supported by burn-onnx will cause `onnx2burn` to fail, which
   the backend correctly maps to `BackendIsNotSupposedToImplementIt` (skip).

3. **Dynamic shapes**: The generated Rust code uses static tensor ranks. If the ONNX test provides
   inputs whose rank differs from what the model declares, inference will fail. This is handled
   gracefully as a skip.

4. **ScalarNative inputs**: Some ONNX models have scalar inputs that burn-onnx converts to native
   Rust types (i64, f32) instead of tensors. The backend handles this by parsing the forward()
   signature, but edge cases may exist.

5. **No PyPI package**: burn-onnx is a Rust project. The Docker image builds from source. The
   `core_packages` in config.json lists `burn-onnx` but there is no pip package; version tracking
   would use the git rev instead.

6. **Serial compilation**: The runner template is shared, so models must be compiled and run
   sequentially (no parallel test execution). The scoreboard harness runs tests serially by default,
   so this is fine.
