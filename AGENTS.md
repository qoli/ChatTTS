# Repository Guidelines

## Project Structure & Module Organization
Core runtime code lives in `ChatTTS/`; `core.py`, `norm.py`, and `model/` implement inference and Velocity streaming. Configurable assets (speaker stats, checksum maps) sit in `ChatTTS/res/` and `ChatTTS/config/`. Reference materials reside under `docs/`, runnable samples and demos sit in `examples/` (CLI, API, ONNX, and notebooks), regression scripts are placed in `tests/`, and helper utilities live in `tools/`.

## Build, Test, and Development Commands
- `pip install --upgrade -r requirements.txt` installs the exact Python dependencies used in CI; pair with `conda create -n chattts python=3.11` when you need isolation.
- `pip install -e .` develops against the package in editable mode, so local changes in `ChatTTS/` are immediately importable.
- `python examples/cmd/run.py --text "Hello" --output out.wav` exercises the default inference path; swap in `examples/cmd/stream.py` for streaming demos.
- `bash tests/testall.sh` runs every file in `tests/*.py` sequentially; keep the script green before opening a PR.

## Coding Style & Naming Conventions
Follow standard Python 3.11 + PyTorch idioms with 4-space indentation and PEP 8 line lengths (~100 chars max). Module names stay snake_case (`ChatTTS/utils/*`), while public classes remain CapWords and constants are SCREAMING_SNAKE_CASE. Formatting commits (`chore(format): run black on dev`) show that `black` is the de facto formatter; run `black .` before committing. Place feature flags and environment lookups near the top of the module, and group imports as standard/lib/third-party/local.

## Testing Guidelines
Existing checks are plain Python scripts invoked via `tests/testall.sh`. Add new coverage by dropping a file such as `tests/test_velocity_tokens.py` and ensuring it executes quickly with deterministic seeds. Prefer `unittest` or bare assertions; skip heavy audio exports in tests unless mocked. When failures involve external weights, point tests to lightweight fixtures under `ChatTTS/res/` so CI remains fast.

## Commit & Pull Request Guidelines
Recent history uses conventional prefixes (`feat:`, `fix:`, `chore(format):`) plus optional scopes and PR numbers, so mirror that style (e.g., `fix(utils): handle missing CUDA`). Each PR should describe the user-facing impact, note dependency updates, link related issues, and attach short logs or sample audio paths if behavior changes.

## Model Assets & Security Tips
Keep downloaded checkpoints in paths referenced by `ChatTTS/res/sha256_map.json` and verify integrity with `sha256sum` before sharing. Avoid committing proprietary voices or API keys; instead document required environment variables in the PR text. When exposing demos (API or Web UI), ensure rate limiting and authentication are toggled on in `examples/api/*.py` before deploying.
