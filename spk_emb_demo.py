#!/usr/bin/env python3
"""Minimal script to run ChatTTS with a fixed speaker embedding."""

import os
import sys
from pathlib import Path

import torch
import soundfile as sf

import ChatTTS

# Customize these constants directly in the file when you need different inputs.
SPEAKER_PATH = Path("pt_file/seed_2279_restored_emb.pt")
TEXT_FILE = Path("input_texts.txt")
OUTPUT_FILE = Path("spk_controlled.wav")


def main() -> None:
    if sys.platform == "darwin":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    if not SPEAKER_PATH.exists():
        raise FileNotFoundError(f"Speaker embedding file not found: {SPEAKER_PATH}")

    print(f"[INFO] Loading speaker embedding from {SPEAKER_PATH}")
    spk = torch.load(SPEAKER_PATH, map_location="cpu")

    print("[INFO] Initializing ChatTTS")
    chat = ChatTTS.Chat()
    if not chat.load(source="local"):
        raise RuntimeError("Failed to load ChatTTS weights from local asset directory.")

    if not TEXT_FILE.exists():
        raise FileNotFoundError(f"Input text file not found: {TEXT_FILE}")
    with open(TEXT_FILE, "r", encoding="utf-8") as handle:
        texts = [
            line.strip()
            for line in handle.readlines()
            if line.strip() and not line.startswith("#")
        ]
    if not texts:
        raise ValueError(f"No valid text lines found in {TEXT_FILE}.")

    print("[INFO] Running inference with custom speaker embedding")
    params = ChatTTS.Chat.InferCodeParams(spk_emb=spk)
    wavs = chat.infer(
        texts,
        skip_refine_text=True,
        params_infer_code=params,
    )

    audio = wavs[0]
    sf.write(str(OUTPUT_FILE), audio, 24000)
    print(f"[INFO] Saved audio to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
