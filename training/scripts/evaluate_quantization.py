#!/usr/bin/env python3
"""
Evaluate FP32 vs INT8 model accuracy on a deterministic validation set.

This script produces a summary for quantization-quality claims by comparing
classification accuracy between:
- Float Keras model (.keras)
- Fully quantized TFLite model (.tflite)

Output: JSON summary with pass/fail gate for maximum allowed degradation.
"""

from __future__ import annotations

import argparse
import json
import random
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np
from dataset_paths import resolve_dataset_dir

try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf = None  # type: ignore[assignment]

try:
    from scipy.io import wavfile
except ModuleNotFoundError:
    wavfile = None  # type: ignore[assignment]

SAMPLE_RATE = 16000
NUM_SAMPLES = 16000
FFT_SIZE = 512
FRAME_STEP = 320
NUM_MEL_BINS = 40
EPSILON = 1e-6
DSP_BACKEND = os.getenv("DSP_BACKEND", "cmsis").lower()

# Label mapping is aligned with firmware inference assumptions:
# class 0 = silence/background, class 1 = unknown, class 2 = target class
LABEL_SILENCE = 0
LABEL_UNKNOWN = 1
LABEL_TARGET = 2

EvalItem = Union[Path, np.ndarray]

try:
    from cmsis_host_dsp import extract_logmel_cmsis
except Exception:
    extract_logmel_cmsis = None


@dataclass
class EvalSample:
    item: EvalItem
    label: int


def _require_deps() -> None:
    missing = []
    if tf is None:
        missing.append("tensorflow")
    if wavfile is None:
        missing.append("scipy")
    if missing:
        raise SystemExit(
            "Missing dependencies: "
            + ", ".join(missing)
            + ". Install with: pip install -r requirements.txt"
        )


def load_audio(item: EvalItem) -> np.ndarray:
    """Load, mono-convert, and length-normalize to 1 second."""
    if isinstance(item, np.ndarray):
        audio = item.astype(np.float32)
    else:
        _, raw = wavfile.read(item)
        audio = raw.astype(np.float32)
        if raw.dtype != np.float32:
            audio = audio / 32768.0

    if audio.ndim > 1:
        audio = audio[:, 0]

    if len(audio) < NUM_SAMPLES:
        audio = np.pad(audio, (0, NUM_SAMPLES - len(audio)), mode="constant")
    else:
        audio = audio[:NUM_SAMPLES]

    return audio.astype(np.float32)


def _extract_features_tf(audio: np.ndarray) -> np.ndarray:
    """Match training/firmware preprocessing: STFT -> mel -> log."""
    audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
    stft = tf.signal.stft(
        audio_tensor,
        frame_length=FFT_SIZE,
        frame_step=FRAME_STEP,
        fft_length=FFT_SIZE,
        window_fn=lambda frame_length, dtype: tf.signal.hann_window(
            frame_length, periodic=True, dtype=dtype
        ),
    )
    spectrogram = tf.abs(stft)

    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=NUM_MEL_BINS,
        num_spectrogram_bins=stft.shape[-1],
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=20.0,
        upper_edge_hertz=4000.0,
    )
    mel = tf.tensordot(spectrogram, mel_matrix, 1)
    mel.set_shape(spectrogram.shape[:-1].concatenate(mel_matrix.shape[-1:]))
    log_mel = tf.math.log(mel + EPSILON)
    return tf.expand_dims(log_mel, axis=-1).numpy().astype(np.float32)


def extract_features(audio: np.ndarray) -> np.ndarray:
    if DSP_BACKEND == "cmsis":
        if extract_logmel_cmsis is None:
            raise RuntimeError("DSP_BACKEND=cmsis requested but cmsis_host_dsp is unavailable")
        return extract_logmel_cmsis(np.asarray(audio, dtype=np.float32), add_channel=True)
    return _extract_features_tf(audio)


def _sample_paths(paths: Sequence[Path], limit: int, rng: random.Random) -> List[Path]:
    if len(paths) <= limit:
        return list(paths)
    return rng.sample(list(paths), limit)


def _load_background_chunks(background_wavs: Sequence[Path], max_items: int, rng: random.Random) -> List[np.ndarray]:
    chunks: List[np.ndarray] = []
    for wav_path in background_wavs:
        _, raw = wavfile.read(wav_path)
        audio = raw.astype(np.float32)
        if raw.dtype != np.float32:
            audio = audio / 32768.0
        if audio.ndim > 1:
            audio = audio[:, 0]

        for start in range(0, max(1, len(audio) - NUM_SAMPLES + 1), NUM_SAMPLES):
            chunk = audio[start:start + NUM_SAMPLES]
            if len(chunk) < NUM_SAMPLES:
                chunk = np.pad(chunk, (0, NUM_SAMPLES - len(chunk)), mode="constant")
            chunks.append(chunk.astype(np.float32))
            if len(chunks) >= max_items * 3:
                break
        if len(chunks) >= max_items * 3:
            break

    if len(chunks) <= max_items:
        return chunks
    return rng.sample(chunks, max_items)


def build_eval_samples(dataset_dir: Path, target_word: str, per_class_limit: int, seed: int) -> List[EvalSample]:
    rng = random.Random(seed)

    validation_list = dataset_dir / "validation_list.txt"
    positives: List[Path] = []
    unknowns: List[Path] = []

    if validation_list.exists():
        for rel in validation_list.read_text(encoding="utf-8").splitlines():
            rel = rel.strip()
            if not rel:
                continue
            p = dataset_dir / rel
            if not p.exists():
                continue
            parts = rel.split("/")
            if not parts:
                continue
            label_dir = parts[0]
            if label_dir == target_word:
                positives.append(p)
            elif label_dir != "_background_noise_":
                unknowns.append(p)
    else:
        positives = list((dataset_dir / target_word).glob("*.wav"))
        for folder in dataset_dir.iterdir():
            if not folder.is_dir() or folder.name in {target_word, "_background_noise_"}:
                continue
            unknowns.extend(folder.glob("*.wav"))

    bg_dir = dataset_dir / "_background_noise_"
    background_wavs = list(bg_dir.glob("*.wav")) if bg_dir.exists() else []

    positives = _sample_paths(positives, per_class_limit, rng)
    unknowns = _sample_paths(unknowns, per_class_limit, rng)
    silences = _load_background_chunks(background_wavs, per_class_limit, rng)

    samples: List[EvalSample] = []
    samples.extend(EvalSample(item=p, label=LABEL_TARGET) for p in positives)
    samples.extend(EvalSample(item=p, label=LABEL_UNKNOWN) for p in unknowns)
    samples.extend(EvalSample(item=chunk, label=LABEL_SILENCE) for chunk in silences)

    rng.shuffle(samples)
    return samples


def build_feature_matrix(samples: Sequence[EvalSample]) -> Tuple[np.ndarray, np.ndarray]:
    features: List[np.ndarray] = []
    labels: List[int] = []
    for sample in samples:
        audio = load_audio(sample.item)
        features.append(extract_features(audio))
        labels.append(sample.label)

    return np.stack(features, axis=0), np.array(labels, dtype=np.int32)


def predict_keras(model: tf.keras.Model, features: np.ndarray, batch_size: int) -> np.ndarray:
    probs = model.predict(features, batch_size=batch_size, verbose=0)
    return np.argmax(probs, axis=1).astype(np.int32)


def _quantize_input(x: np.ndarray, scale: float, zero_point: int, dtype: np.dtype) -> np.ndarray:
    if scale <= 0:
        raise ValueError("Invalid quantization scale in TFLite input tensor")
    q = np.round(x / scale + zero_point)
    if dtype == np.int8:
        q = np.clip(q, -128, 127)
    elif dtype == np.uint8:
        q = np.clip(q, 0, 255)
    return q.astype(dtype)


def _dequantize_output(y: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return (y.astype(np.float32) - float(zero_point)) * float(scale)


def predict_tflite(model_path: Path, features: np.ndarray) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]
    in_dtype = in_detail["dtype"]
    out_dtype = out_detail["dtype"]

    in_scale, in_zp = in_detail["quantization"]
    out_scale, out_zp = out_detail["quantization"]

    preds: List[int] = []
    for i in range(features.shape[0]):
        x = features[i:i + 1]
        if in_dtype in (np.int8, np.uint8):
            x = _quantize_input(x, in_scale, in_zp, in_dtype)
        else:
            x = x.astype(in_dtype)

        interpreter.set_tensor(in_detail["index"], x)
        interpreter.invoke()
        y = interpreter.get_tensor(out_detail["index"])

        if out_dtype in (np.int8, np.uint8):
            y = _dequantize_output(y, out_scale, out_zp)

        preds.append(int(np.argmax(y[0])))

    return np.array(preds, dtype=np.int32)


def accuracy_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred) * 100.0)


def class_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {}
    for label, name in ((LABEL_SILENCE, "silence"), (LABEL_UNKNOWN, "unknown"), (LABEL_TARGET, "target")):
        idx = y_true == label
        if np.any(idx):
            out[name] = float(np.mean(y_true[idx] == y_pred[idx]) * 100.0)
        else:
            out[name] = None
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate quantization accuracy degradation")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--fp32-model", default="training/outputs/model.keras")
    parser.add_argument("--int8-model", default="training/outputs/model.tflite")
    parser.add_argument("--target-word", default="on")
    parser.add_argument("--per-class-limit", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-drop-pp",
        type=float,
        default=2.0,
        help="Maximum allowed INT8 accuracy drop in percentage points",
    )
    parser.add_argument("--output", default="training/outputs/quantization_accuracy_summary.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _require_deps()

    try:
        dataset_dir = resolve_dataset_dir(args.dataset_dir)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
    fp32_model_path = Path(args.fp32_model)
    int8_model_path = Path(args.int8_model)
    output_path = Path(args.output)

    if not dataset_dir.exists():
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")
    if not fp32_model_path.exists():
        raise SystemExit(f"FP32 model not found: {fp32_model_path}")
    if not int8_model_path.exists():
        raise SystemExit(f"INT8 model not found: {int8_model_path}")

    print("Building deterministic evaluation set...")
    samples = build_eval_samples(
        dataset_dir=dataset_dir,
        target_word=args.target_word,
        per_class_limit=args.per_class_limit,
        seed=args.seed,
    )
    if not samples:
        raise SystemExit("No evaluation samples found. Check dataset path and contents.")

    features, labels = build_feature_matrix(samples)
    print(f"Samples: {len(samples)} | Feature shape: {features.shape}")

    print("Running FP32 model...")
    fp32_model = tf.keras.models.load_model(fp32_model_path)
    fp32_preds = predict_keras(fp32_model, features, args.batch_size)

    print("Running INT8 model...")
    int8_preds = predict_tflite(int8_model_path, features)

    fp32_acc = accuracy_percent(labels, fp32_preds)
    int8_acc = accuracy_percent(labels, int8_preds)
    drop_pp = fp32_acc - int8_acc
    passes = drop_pp <= args.max_drop_pp

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "fp32_model": str(fp32_model_path),
        "int8_model": str(int8_model_path),
        "target_word": args.target_word,
        "sample_count": int(len(samples)),
        "per_class_limit": int(args.per_class_limit),
        "metrics": {
            "fp32_accuracy_percent": fp32_acc,
            "int8_accuracy_percent": int8_acc,
            "accuracy_drop_percentage_points": drop_pp,
            "max_allowed_drop_percentage_points": float(args.max_drop_pp),
            "pass": bool(passes),
            "fp32_per_class_accuracy_percent": class_accuracy(labels, fp32_preds),
            "int8_per_class_accuracy_percent": class_accuracy(labels, int8_preds),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Quantization Accuracy Summary ===")
    print(f"FP32 accuracy: {fp32_acc:.2f}%")
    print(f"INT8 accuracy: {int8_acc:.2f}%")
    print(f"Drop: {drop_pp:.2f} pp (limit: {args.max_drop_pp:.2f} pp)")
    print(f"Pass: {passes}")
    print(f"Summary: {output_path}")

    return 0 if passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
