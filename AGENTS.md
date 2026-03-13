# AGENTS.md

This file is the contributor and coding-agent guide for the CamWatch repository.

## Project Overview

CamWatch is a Python 3.14.3 computer-vision application for monitoring video streams. It detects motion, faces, and cars, can record video clips, can publish events to MQTT, and includes optional face recognition and number plate OCR support.

The codebase is a script-driven pipeline rather than a web application or service framework. Most runtime behavior is configured through YAML files.

## Primary Stack

- Python 3.14.3
- OpenCV
- NumPy
- PyYAML
- `face-recognition` / dlib
- `paho-mqtt`
- `pytesseract`

## Repository Layout

- `src/` - application source code
- `src/capture.py` - main capture and monitoring loop
- `src/tasks.py` - image preparation and detector task orchestration
- `src/config.py` - config discovery, defaults, merge behavior, and validation
- `src/face_trainer.py` - face encoding training workflow
- `src/face_recog.py` - face recognition logic
- `src/cascades/` - bundled Haar cascade XML files
- `config/` - YAML configuration templates and defaults
- `docs/` - supporting documentation assets
- `pyproject.toml` - canonical project metadata and uv-managed dependencies
- `requirements.txt` - deprecated compatibility dependency list

## How To Run

Install dependencies:

```bash
uv venv --python 3.14.3
uv sync
```

Optional face-recognition dependencies:

```bash
uv sync --extra face
```

Create local config:

```bash
cp config/camwatch-quick_start.yaml config/camwatch.yaml
```

Start monitoring a configured channel:

```bash
uv run python src/capture.py {channel_name}
```

Train face encodings:

```bash
uv run python src/face_trainer.py
```

## Configuration Rules

- Treat `config/camwatch-defaults.yaml` as the canonical reference for supported settings.
- Runtime config is expected in `camwatch.yaml` or `.camwatch.yaml` and is discovered from multiple paths.
- `CV_CONFIG_PATH` only changes where CamWatch searches for config files; it does not replace the YAML-based configuration model.
- Channel config is merged with `default`, then validated and loaded by `src/config.py`.
- When changing configuration behavior, update both `src/config.py` and the YAML defaults/templates together.

Config search order currently includes:

- `CV_CONFIG_PATH` if set
- home directory
- `~/.camwatch`
- `/etc`
- repo `config/`
- repo root
- current working directory

## Development Notes

- Run scripts from the repository root. Several imports assume repo-root execution with files under `src/`.
- Use `uv` as the default workflow for dependency sync and command execution.
- Treat face recognition as optional on Python 3.14 until a more reliable replacement for `dlib` is adopted.
- `src/capture.py` is the main operational entry point.
- `src/tasks.py` currently routes face detection through `detect_face_dlib.py` rather than the Haar-cascade face detector.
- Face recognition depends on training data plus a generated `face_encodings.dat` file in the configured face-recognition directory.
- MQTT is optional and only active when `services.mqtt.host` is configured.
- Number plate OCR is optional and depends on Tesseract being available on the system.

## Expectations For Code Changes

- Keep changes small and aligned with the current script-based architecture unless performing deliberate refactoring.
- Preserve YAML compatibility when modifying config structures.
- Avoid hardcoding environment-specific absolute paths in defaults, examples, or new logic.
- Prefer improving existing modules over adding parallel implementations without a migration plan.
- If updating detection behavior, review how the change affects motion detection, object detection, video writing, and MQTT events together.

## Verification

There is currently no automated test suite in this repository.

When making changes, use lightweight verification such as:

- `uv venv --python 3.14.3`
- `uv sync`
- `uv run python src/capture.py {channel_name}` with a valid local config and video source
- `uv sync --extra face && uv run python src/face_trainer.py` when changing face-recognition code
- targeted manual validation of YAML loading, motion detection flow, and optional MQTT publishing

If you add tests, document how to run them here and in the README.

## Explicit Warnings

- There are no automated tests, CI workflows, or formal quality gates in the current repo. Do not assume a change is safe without manual validation.
- Native dependencies are significant. `face-recognition` and dlib may require platform-specific build tooling, and OCR features require a working Tesseract installation.
- Python 3.14 is now the default runtime, but face recognition remains an optional workflow because `dlib` compatibility is still fragile.
- The project now uses `uv` and `pyproject.toml` as the source of truth for dependencies. `requirements.txt` is temporary compatibility-only metadata.
- The Dockerfile may still need environment-specific validation because native CV dependencies and runtime device access can vary by host.
- Some defaults and example values still contain machine-specific or legacy assumptions; review carefully before reusing them in new docs or code.
- The codebase mixes active paths and older experiments. Confirm the real execution path before refactoring seemingly duplicated logic.

## Proposed Refactor Roadmap

### Phase 1: Stabilize

- Add a basic automated test harness for config loading, config merging, and non-video utility behavior.
- Add linting and formatting tools, then document the commands.
- Finish the Python 3.14 `uv` migration by validating Docker and local workflows on supported machines.
- Audit defaults and example configs for outdated paths, names, and comments.

### Phase 2: Improve Structure

- Convert `src/` into a proper package to remove repo-root import assumptions.
- Introduce a single CLI entry point for capture, training, and diagnostics.
- Separate runtime pipeline logic from UI/debug display logic.
- Isolate integrations such as MQTT, OCR, and face recognition behind clearer interfaces.

### Phase 3: Increase Reliability

- Add integration tests for config-driven startup and channel selection.
- Add fixtures or sample media for detector smoke tests.
- Improve error handling around missing models, missing encodings, camera failures, and invalid config.
- Replace print-based operational logging with structured logging.

### Phase 4: Modernize Features

- Revisit the detector pipeline and standardize which face/car detectors are active and supported.
- Add a clearer plugin or service model for optional features.
- Improve documentation for local development, deployment, and Home Assistant integration.
- Evaluate performance, thread safety, and resource cleanup across long-running capture sessions.

## Good Contributor Workflow

1. Read `README.md`, `config/camwatch-defaults.yaml`, and `src/config.py` before changing runtime behavior.
2. Make focused changes in `src/` and keep config/docs in sync.
3. Validate with manual script runs because automated coverage does not yet exist.
4. Document any new command, dependency, or workflow change in both `README.md` and this file.
