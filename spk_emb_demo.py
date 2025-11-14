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
REFINE_PROMPT = "[oral_2][laugh_0][break_4]"


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
        texts = []
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            processed = line.replace("\n", "").strip()
            if processed:
                texts.append(processed)
    if not texts:
        raise ValueError(f"No valid text lines found in {TEXT_FILE}.")

    print("[INFO] Running inference with custom speaker embedding")
    params = ChatTTS.Chat.InferCodeParams(
        spk_emb=spk,
        temperature=0.001,
        top_P=0.7,
        top_K=20,
    )
    refine_params = ChatTTS.Chat.RefineTextParams(prompt=REFINE_PROMPT)
    wavs = chat.infer(
        texts,
        params_refine_text=refine_params,
        params_infer_code=params,
    )

    suffix = OUTPUT_FILE.suffix or ".wav"
    base_name = OUTPUT_FILE.stem if OUTPUT_FILE.suffix else OUTPUT_FILE.name or "spk_output"
    parent = OUTPUT_FILE.parent if OUTPUT_FILE.parent != Path("") else Path(".")
    parent.mkdir(parents=True, exist_ok=True)
    targets = (
        [OUTPUT_FILE]
        if len(wavs) == 1
        else [parent / f"{base_name}_{idx}{suffix}" for idx in range(len(wavs))]
    )

    for wav, path in zip(wavs, targets):
        sf.write(path, wav, 24000)
        print(f"[INFO] Saved audio to {path}")


if __name__ == "__main__":
    main()
