#!/usr/bin/env python3
"""Light-weight smoke test for ChatTTS WebUI dependencies and assets."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple


REPO_ROOT = Path(__file__).resolve().parent
REQUIRED_MODULES = [
    "torch",
    "torchaudio",
    "gradio",
    "av",
    "pybase16384",
    "vector_quantize_pytorch",
    "vocos",
]


def check_modules() -> List[Tuple[str, bool, str]]:
    """Try importing all dependencies so we can fail fast with actionable info."""
    statuses: List[Tuple[str, bool, str]] = []
    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            statuses.append((module_name, False, repr(exc)))
        else:
            statuses.append((module_name, True, ""))
    return statuses


def ensure_chattts(model_root: Path, text: str, output: Path, speaker: str | None):
    """Instantiate ChatTTS, load local weights, and optionally synthesize audio."""
    import numpy as np
    import torch
    import torchaudio

    try:
        import ChatTTS
        from ChatTTS.utils import check_all_assets, logger as utils_logger
    except Exception as exc:  # noqa: BLE001
        logging.error("Failed to import ChatTTS: %s", exc)
        return False

    utils_logger.set_logger(logging.getLogger("ChatTTS"))

    sha_map_path = REPO_ROOT / "ChatTTS" / "res" / "sha256_map.json"
    with open(sha_map_path, "r", encoding="utf-8") as handle:
        sha_map = json.load(handle)

    logging.info("Checking weight files under %s", model_root)
    if not check_all_assets(model_root, sha_map, update=False):
        logging.error(
            "Missing weights in %s/asset. "
            "Download them manually or adjust --model-root.",
            model_root,
        )
        return False

    chat_logger = logging.getLogger("ChatTTS")
    chat = ChatTTS.Chat(chat_logger)
    logging.info("Loading ChatTTS from %s", model_root)
    if not chat.load(source="custom", custom_path=str(model_root)):
        logging.error("ChatTTS load failed even though files were detected.")
        return False

    spk = speaker or chat.sample_random_speaker()
    logging.info("Generating preview audio with sampled speaker.")

    wavs = chat.infer(
        [text],
        params_infer_code=ChatTTS.Chat.InferCodeParams(spk_emb=spk),
    )
    audio = torch.from_numpy(np.asarray(wavs[0], dtype=np.float32)).unsqueeze(0)
    torchaudio.save(str(output), audio, 24000)
    logging.info("Smoke test audio saved to %s", output)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate ChatTTS WebUI prerequisites. "
            "It first checks Python dependencies, then tries a tiny inference "
            "using local weights if they are present."
        )
    )
    parser.add_argument(
        "--model-root",
        default=str(REPO_ROOT),
        help="Path whose asset/ folder already contains the ChatTTS weights.",
    )
    parser.add_argument(
        "--text",
        default="ChatTTS smoke test sample.",
        help="Text used for the optional synthesis run.",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "smoke_test.wav"),
        help="Output wav file for the synthesized preview.",
    )
    parser.add_argument(
        "--speaker",
        default=None,
        help="Optional speaker embedding string to reuse. Defaults to a random pick.",
    )
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Skip model loading and synthesis, only validate Python modules.",
    )
    args = parser.parse_args()

    if sys.platform == "darwin":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stdout
    )
    logging.info("Running ChatTTS environment smoke test.")

    module_statuses = check_modules()
    missing = [module for module, ok, _ in module_statuses if not ok]
    for module_name, ok, err in module_statuses:
        if ok:
            logging.info("[OK] %s", module_name)
        else:
            logging.error("[FAIL] %s (%s)", module_name, err)

    if missing:
        logging.error(
            "Install the missing modules above (e.g. pip install -r requirements.txt) "
            "before retrying."
        )
        return 1

    if args.skip_audio:
        logging.info("Dependency check completed. Skipping audio generation.")
        return 0

    model_root = Path(args.model_root).resolve()
    output_path = Path(args.output).resolve()

    if ensure_chattts(model_root, args.text, output_path, args.speaker):
        logging.info("Smoke test finished successfully.")
        return 0

    logging.error("Smoke test failed. See logs above for details.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
