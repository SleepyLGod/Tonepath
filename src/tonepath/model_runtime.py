"""Workspace-local model runtime management for Tonepath."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from tonepath import config


ESSENTIA_TF_RUNTIME = "essentia-tf-py311"
ESSENTIA_TF_PACKAGE = "essentia-tensorflow==2.1b6.dev1389"
ESSENTIA_MODEL_BASE_URL = "https://essentia.upf.edu/models"

ESSENTIA_MODEL_FILES = (
    (
        "discogs-effnet-bs64-1.pb",
        f"{ESSENTIA_MODEL_BASE_URL}/music-style-classification/discogs-effnet/discogs-effnet-bs64-1.pb",
    ),
    (
        "voice_instrumental-musicnn-msd-1.pb",
        f"{ESSENTIA_MODEL_BASE_URL}/classifiers/voice_instrumental/voice_instrumental-musicnn-msd-1.pb",
    ),
    (
        "voice_instrumental-musicnn-msd-1.json",
        f"{ESSENTIA_MODEL_BASE_URL}/classifiers/voice_instrumental/voice_instrumental-musicnn-msd-1.json",
    ),
    (
        "mtg_jamendo_genre-discogs-effnet-1.pb",
        f"{ESSENTIA_MODEL_BASE_URL}/classification-heads/mtg_jamendo_genre/mtg_jamendo_genre-discogs-effnet-1.pb",
    ),
    (
        "mtg_jamendo_genre-discogs-effnet-1.json",
        f"{ESSENTIA_MODEL_BASE_URL}/classification-heads/mtg_jamendo_genre/mtg_jamendo_genre-discogs-effnet-1.json",
    ),
    (
        "mtg_jamendo_moodtheme-discogs-effnet-1.pb",
        f"{ESSENTIA_MODEL_BASE_URL}/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.pb",
    ),
    (
        "mtg_jamendo_moodtheme-discogs-effnet-1.json",
        f"{ESSENTIA_MODEL_BASE_URL}/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.json",
    ),
    (
        "mtg_jamendo_instrument-discogs-effnet-1.pb",
        f"{ESSENTIA_MODEL_BASE_URL}/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-discogs-effnet-1.pb",
    ),
    (
        "mtg_jamendo_instrument-discogs-effnet-1.json",
        f"{ESSENTIA_MODEL_BASE_URL}/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-discogs-effnet-1.json",
    ),
)


def runtime_root() -> Path:
    """Return the root directory for all isolated runtime assets."""

    return config.ensure_data_dir() / "runtimes"


def uv_python_install_dir() -> Path:
    """Return the isolated uv-managed Python install directory."""

    return runtime_root() / "python"


def uv_cache_dir() -> Path:
    """Return the isolated uv cache directory."""

    return config.ensure_data_dir() / "cache" / "uv"


def pip_cache_dir() -> Path:
    """Return the isolated pip cache directory."""

    return config.ensure_data_dir() / "cache" / "pip"


def python_userbase_dir() -> Path:
    """Return the isolated Python userbase directory."""

    return runtime_root() / "python-user"


def isolated_runtime_env() -> dict[str, str]:
    """Return environment variables that keep setup inside Tonepath data dir."""

    data_root = config.ensure_data_dir()
    env = os.environ.copy()
    env.update(
        {
            "TONEPATH_HOME": str(data_root),
            "UV_PYTHON_INSTALL_DIR": str(uv_python_install_dir()),
            "UV_CACHE_DIR": str(uv_cache_dir()),
            "PIP_CACHE_DIR": str(pip_cache_dir()),
            "PYTHONUSERBASE": str(python_userbase_dir()),
            "PYTHONNOUSERSITE": "1",
            "PIP_NO_INPUT": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        }
    )
    return env


def ensure_isolation_dirs() -> None:
    """Create all local runtime/cache directories before setup."""

    for path in (uv_python_install_dir(), uv_cache_dir(), pip_cache_dir(), python_userbase_dir(), runtime_dir(), essentia_model_dir()):
        path.mkdir(parents=True, exist_ok=True)


def isolation_report() -> str:
    """Return the active local runtime/cache directory plan."""

    return "\n".join(
        [
            "Tonepath isolated runtime paths",
            f"TONEPATH_HOME={config.ensure_data_dir()}",
            f"UV_PYTHON_INSTALL_DIR={uv_python_install_dir()}",
            f"UV_CACHE_DIR={uv_cache_dir()}",
            f"PIP_CACHE_DIR={pip_cache_dir()}",
            f"PYTHONUSERBASE={python_userbase_dir()}",
        ]
    )


@dataclass(frozen=True)
class RuntimeStatus:
    """Status for one local model runtime."""

    runtime_dir: Path
    python: Path
    runner: Path
    model_dir: Path
    ready: bool
    missing: tuple[str, ...]


def runtime_dir() -> Path:
    """Return the workspace-local Essentia TensorFlow runtime directory."""

    return config.ensure_data_dir() / "runtimes" / ESSENTIA_TF_RUNTIME


def runtime_python() -> Path:
    """Return the runtime Python executable path."""

    root = runtime_dir()
    if sys.platform == "win32":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def runner_path() -> Path:
    """Return the generated worker script path."""

    return runtime_dir() / "tonepath_essentia_tf_worker.py"


def essentia_model_dir() -> Path:
    """Return the workspace-local Essentia model directory."""

    return config.ensure_data_dir() / "cache" / "models" / "essentia"


def setup_essentia_tf_runtime() -> RuntimeStatus:
    """Create the local Essentia TensorFlow runtime and download model files."""

    ensure_isolation_dirs()
    python311 = ensure_isolated_python311()
    root = runtime_dir()
    if not runtime_python().exists():
        subprocess.run([str(python311), "-m", "venv", str(root)], check=True, env=isolated_runtime_env())
    subprocess.run([str(runtime_python()), "-m", "pip", "install", "--upgrade", "pip"], check=True, env=isolated_runtime_env())
    subprocess.run([str(runtime_python()), "-m", "pip", "install", ESSENTIA_TF_PACKAGE], check=True, env=isolated_runtime_env())

    write_runner()
    download_essentia_models()
    return model_runtime_status()


def ensure_isolated_python311() -> Path:
    """Install or find uv-managed Python 3.11 inside the local runtime root."""

    existing = find_isolated_python311()
    if existing is not None:
        return existing
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("Essentia TensorFlow setup requires uv on PATH.")
    subprocess.run([uv, "python", "install", "3.11"], check=True, env=isolated_runtime_env())
    existing = find_isolated_python311()
    if existing is None:
        raise RuntimeError("uv installed Python 3.11, but no isolated python executable was found.")
    return existing


def find_isolated_python311() -> Path | None:
    """Return an isolated Python 3.11 executable if it exists."""

    root = uv_python_install_dir()
    candidates = sorted(root.glob("*/bin/python3.11")) + sorted(root.glob("*/bin/python"))
    for candidate in candidates:
        if not candidate.exists():
            continue
        completed = subprocess.run(
            [str(candidate), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            check=False,
            capture_output=True,
            text=True,
            env=isolated_runtime_env(),
        )
        if completed.returncode == 0 and completed.stdout.strip() == "3.11":
            return candidate
    return None


def write_runner() -> Path:
    """Write the Essentia TensorFlow worker script into the model runtime."""

    path = runner_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ESSENTIA_TF_WORKER, encoding="utf-8")
    return path


def download_essentia_models() -> None:
    """Download missing Essentia model files into the local model cache."""

    model_dir = essentia_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in ESSENTIA_MODEL_FILES:
        target = model_dir / filename
        if target.exists() and target.stat().st_size > 0:
            continue
        urllib.request.urlretrieve(url, target)


def model_runtime_status() -> RuntimeStatus:
    """Return current Essentia TensorFlow runtime readiness."""

    missing: list[str] = []
    python = runtime_python()
    runner = runner_path()
    model_dir = essentia_model_dir()
    if not python.exists():
        missing.append("runtime python")
    if not runner.exists():
        missing.append("worker script")
    for filename, _url in ESSENTIA_MODEL_FILES:
        if not (model_dir / filename).exists():
            missing.append(filename)
    return RuntimeStatus(
        runtime_dir=runtime_dir(),
        python=python,
        runner=runner,
        model_dir=model_dir,
        ready=not missing,
        missing=tuple(missing),
    )


def model_runtime_report() -> str:
    """Return a user-facing report for local model runtime status."""

    status = model_runtime_status()
    lines = [
        "Tonepath model runtime doctor",
        f"Data directory: {config.ensure_data_dir()}",
        f"UV Python install dir: {uv_python_install_dir()}",
        f"UV cache dir: {uv_cache_dir()}",
        f"PIP cache dir: {pip_cache_dir()}",
        f"Python userbase: {python_userbase_dir()}",
        f"Runtime directory: {status.runtime_dir}",
        f"Runtime python: {status.python} ({'ok' if status.python.exists() else 'missing'})",
        f"Worker script: {status.runner} ({'ok' if status.runner.exists() else 'missing'})",
        f"Model directory: {status.model_dir}",
        f"Ready: {status.ready}",
    ]
    if status.missing:
        lines.append("Missing:")
        lines.extend(f"  {item}" for item in status.missing)
        lines.append("Run: uv run tonepath models setup essentia-tf")
    return "\n".join(lines)


def ensure_essentia_tf_runtime() -> RuntimeStatus:
    """Return runtime status or raise a setup hint when unavailable."""

    status = model_runtime_status()
    if not status.ready:
        raise RuntimeError("Essentia TensorFlow tagging runtime is not ready. Run: uv run tonepath models setup essentia-tf")
    return status


def run_essentia_tf_tags(path: Path) -> dict[str, object]:
    """Run the local Essentia TensorFlow worker for one audio file."""

    status = ensure_essentia_tf_runtime()
    completed = subprocess.run(
        [str(status.python), str(status.runner), str(path), str(status.model_dir)],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "Essentia TensorFlow tagging failed."
        raise RuntimeError(message)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Essentia TensorFlow worker returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Essentia TensorFlow worker returned an invalid payload.")
    return payload


ESSENTIA_TF_WORKER = r'''
"""Tonepath Essentia TensorFlow worker."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from essentia.standard import MonoLoader, TensorflowPredict2D, TensorflowPredictEffnetDiscogs, TensorflowPredictMusiCNN


EFFNET_HEADS = (
    "mtg_jamendo_genre-discogs-effnet-1",
    "mtg_jamendo_moodtheme-discogs-effnet-1",
    "mtg_jamendo_instrument-discogs-effnet-1",
)


def main() -> None:
    audio_path = Path(sys.argv[1])
    model_dir = Path(sys.argv[2])
    audio = MonoLoader(filename=str(audio_path), sampleRate=16000, resampleQuality=4)()
    embeddings = TensorflowPredictEffnetDiscogs(
        graphFilename=str(model_dir / "discogs-effnet-bs64-1.pb"),
        output="PartitionedCall:1",
    )(audio)

    tags = []
    voice_metadata = metadata_for(model_dir / "voice_instrumental-musicnn-msd-1.json")
    voice_predictions = TensorflowPredictMusiCNN(
        graphFilename=str(model_dir / "voice_instrumental-musicnn-msd-1.pb"),
        output=prediction_output(voice_metadata),
    )(audio)
    voice_scores = np.asarray(voice_predictions).mean(axis=0)
    voice_labels = labels_from_metadata(voice_metadata)
    voice_score = score_for_label(voice_scores, voice_labels, "voice")
    instrumental_score = score_for_label(voice_scores, voice_labels, "instrumental")
    for index, score in top_scores(voice_scores, voice_labels, 2):
        tags.append([voice_labels[index], float(score)])

    for head in EFFNET_HEADS:
        metadata = metadata_for(model_dir / f"{head}.json")
        predictions = TensorflowPredict2D(
            graphFilename=str(model_dir / f"{head}.pb"),
            output=prediction_output(metadata),
        )(embeddings)
        scores = np.asarray(predictions).mean(axis=0)
        labels = labels_from_metadata(metadata)
        for index, score in top_scores(scores, labels, 5):
            label = labels[index]
            value = float(score)
            tags.append([label, value])

    vocalness = None
    if voice_score is not None and instrumental_score is not None:
        total = voice_score + instrumental_score
        vocalness = None if total <= 0 else voice_score / total
    elif voice_score is not None:
        vocalness = voice_score

    print(json.dumps({"vocalness": vocalness, "tags": tags}, ensure_ascii=False))


def metadata_for(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def labels_from_metadata(data: dict) -> list[str]:
    labels = data.get("classes") or data.get("labels")
    if not isinstance(labels, list):
        raise RuntimeError("Missing model classes")
    return [str(label) for label in labels]


def prediction_output(data: dict) -> str:
    for output in data.get("schema", {}).get("outputs", []):
        if output.get("output_purpose") == "predictions":
            name = output.get("name")
            if name:
                return str(name)
    return "model/Sigmoid"


def score_for_label(scores: np.ndarray, labels: list[str], wanted: str) -> float | None:
    for index, label in enumerate(labels):
        if label.lower() == wanted:
            return float(scores[index])
    return None


def top_scores(scores: np.ndarray, labels: list[str], limit: int) -> list[tuple[int, float]]:
    count = min(limit, len(labels), len(scores))
    indices = np.argsort(scores)[::-1][:count]
    return [(int(index), float(scores[index])) for index in indices]


if __name__ == "__main__":
    main()
'''
