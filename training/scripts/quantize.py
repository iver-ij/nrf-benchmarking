import os
import random
import shutil
from pathlib import Path
from typing import Iterable, List, Sequence, Union

import numpy as np
import tensorflow as tf
from scipy.io import wavfile
from dataset_paths import resolve_dataset_dir

MODEL_KERAS_PATH = Path("training/outputs/model.keras")
TFLITE_FILE = Path("training/outputs/model.tflite")
C_MODEL_FILE = Path("firmware/src/model.cpp")

SAMPLE_RATE = 16000
NUM_SAMPLES = 16000
FFT_SIZE = 512
FRAME_STEP = 320
NUM_MEL_BINS = 40
EPSILON = 1e-6
DSP_BACKEND = os.getenv("DSP_BACKEND", "cmsis").lower()

TARGET_WORD = os.getenv("TARGET_WORD", "on")
RNG_SEED = int(os.getenv("REP_SEED", "42"))
# PTQ calibration quality depends heavily on input distribution.
# Use deterministic, class-balanced representative set with tunable mix.
REP_TARGET_SAMPLES = int(os.getenv("REP_TARGET_SAMPLES", "450"))
REP_UNKNOWN_SAMPLES = int(os.getenv("REP_UNKNOWN_SAMPLES", "450"))
REP_SILENCE_SAMPLES = int(os.getenv("REP_SILENCE_SAMPLES", "100"))
USE_NEW_QUANTIZER = os.getenv("USE_NEW_QUANTIZER", "1") != "0"

EvalItem = Union[Path, np.ndarray]


try:
    from cmsis_host_dsp import extract_logmel_cmsis
except Exception:
    extract_logmel_cmsis = None


def print_configuration(dataset_dir: Path) -> None:
    print("Configuration")
    print(f"Model Input: {MODEL_KERAS_PATH.resolve()}")
    print(f"Dataset Dir: {dataset_dir}")
    print(f"TFLite Output: {TFLITE_FILE.resolve()}")
    print(
        "PTQ Calibration: "
        f"target={REP_TARGET_SAMPLES} unknown={REP_UNKNOWN_SAMPLES} "
        f"silence={REP_SILENCE_SAMPLES} seed={RNG_SEED} "
        f"new_quantizer={USE_NEW_QUANTIZER}"
    )


def load_audio(item: EvalItem) -> np.ndarray | None:
    try:
        if isinstance(item, np.ndarray):
            data = item.astype(np.float32)
        else:
            _, raw = wavfile.read(item)
            data = raw.astype(np.float32)
            if raw.dtype != np.float32:
                data = data / 32768.0

        if data.ndim > 1:
            data = data[:, 0]

        if len(data) < NUM_SAMPLES:
            data = np.pad(data, (0, NUM_SAMPLES - len(data)), mode="constant")
        else:
            data = data[:NUM_SAMPLES]

        return data.astype(np.float32)
    except Exception:
        return None


def _extract_features_for_calibration_tf(audio_data: np.ndarray) -> np.ndarray:
    audio_tensor = tf.convert_to_tensor(audio_data, dtype=tf.float32)
    stft = tf.signal.stft(
        audio_tensor,
        frame_length=FFT_SIZE,
        frame_step=FRAME_STEP,
        fft_length=FFT_SIZE,
        # Make window definition explicit to avoid TF version defaults drifting.
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
    return tf.reshape(log_mel, (1, 49, 40, 1)).numpy().astype(np.float32)


def extract_features_for_calibration(audio_data: np.ndarray) -> np.ndarray:
    if DSP_BACKEND == "cmsis":
        if extract_logmel_cmsis is None:
            raise RuntimeError("DSP_BACKEND=cmsis requested but cmsis_host_dsp is unavailable")
        feature = extract_logmel_cmsis(np.asarray(audio_data, dtype=np.float32), add_channel=True)
        return np.expand_dims(feature, axis=0).astype(np.float32)
    return _extract_features_for_calibration_tf(audio_data)


def _sample_paths(paths: Sequence[Path], limit: int, rng: random.Random) -> List[Path]:
    paths = list(paths)
    if len(paths) <= limit:
        return paths
    return rng.sample(paths, limit)


def _load_background_chunks(
    background_wavs: Sequence[Path], max_items: int, rng: random.Random
) -> List[np.ndarray]:
    chunks: List[np.ndarray] = []
    for wav_path in background_wavs:
        try:
            _, raw = wavfile.read(wav_path)
        except Exception:
            continue

        audio = raw.astype(np.float32)
        if raw.dtype != np.float32:
            audio = audio / 32768.0
        if audio.ndim > 1:
            audio = audio[:, 0]

        # 50% overlap to improve dynamic range coverage for calibration.
        hop = NUM_SAMPLES // 2
        if len(audio) < NUM_SAMPLES:
            audio = np.pad(audio, (0, NUM_SAMPLES - len(audio)), mode="constant")

        for start in range(0, max(1, len(audio) - NUM_SAMPLES + 1), hop):
            chunk = audio[start : start + NUM_SAMPLES]
            if len(chunk) < NUM_SAMPLES:
                chunk = np.pad(chunk, (0, NUM_SAMPLES - len(chunk)), mode="constant")
            chunks.append(chunk.astype(np.float32))
            if len(chunks) >= max_items * 4:
                break
        if len(chunks) >= max_items * 4:
            break

    if len(chunks) <= max_items:
        return chunks
    return rng.sample(chunks, max_items)


def build_representative_items(dataset_dir: Path) -> List[EvalItem]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found at {dataset_dir}")

    rng = random.Random(RNG_SEED)

    target_paths = list((dataset_dir / TARGET_WORD).glob("*.wav"))

    unknown_paths: List[Path] = []
    for folder in dataset_dir.iterdir():
        if not folder.is_dir() or folder.name in {TARGET_WORD, "_background_noise_"}:
            continue
        unknown_paths.extend(folder.glob("*.wav"))

    background_wavs = list((dataset_dir / "_background_noise_").glob("*.wav"))

    target_items = _sample_paths(target_paths, REP_TARGET_SAMPLES, rng)
    unknown_items = _sample_paths(unknown_paths, REP_UNKNOWN_SAMPLES, rng)
    silence_items = _load_background_chunks(background_wavs, REP_SILENCE_SAMPLES, rng)

    items: List[EvalItem] = []
    items.extend(target_items)
    items.extend(unknown_items)
    items.extend(silence_items)
    rng.shuffle(items)

    print(
        "Representative set: "
        f"target={len(target_items)} "
        f"unknown={len(unknown_items)} "
        f"silence={len(silence_items)} "
        f"total={len(items)}"
    )

    if not items:
        raise RuntimeError("Representative dataset is empty")

    return items


def representative_dataset(dataset_dir: Path) -> Iterable[list[np.ndarray]]:
    items = build_representative_items(dataset_dir)
    emitted = 0
    for item in items:
        audio = load_audio(item)
        if audio is None:
            continue
        input_tensor = extract_features_for_calibration(audio)
        emitted += 1
        if emitted % 100 == 0:
            print(f"Collected calibration samples: {emitted}")
        yield [input_tensor]

    print(f"Calibration samples emitted: {emitted}")


def hex_to_c_array(data: bytes, var_name: str) -> str:
    c_str = ""
    c_str += "#include <model.h>\n\n"
    c_str += f"const unsigned char {var_name}[] __attribute__((aligned(16))) = {{\n"
    for i, val in enumerate(data):
        c_str += f"0x{val:02x}, "
        if (i + 1) % 12 == 0:
            c_str += "\n"
    c_str += "};\n\n"
    c_str += f"const unsigned int {var_name}_len = sizeof({var_name});\n"
    return c_str


def quantize() -> None:
    if not MODEL_KERAS_PATH.exists():
        print(f"Model file missing at {MODEL_KERAS_PATH}")
        return

    dataset_dir = resolve_dataset_dir()
    print_configuration(dataset_dir)

    model = tf.keras.models.load_model(MODEL_KERAS_PATH)

    temp_dir = Path("training/outputs/temp_quant_model")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    try:
        model.export(temp_dir)

        converter = tf.lite.TFLiteConverter.from_saved_model(str(temp_dir))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        converter.representative_dataset = lambda: representative_dataset(dataset_dir)
        converter.experimental_new_quantizer = USE_NEW_QUANTIZER

        tflite_model = converter.convert()
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    TFLITE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with TFLITE_FILE.open("wb") as f:
        f.write(tflite_model)

    print(f"\nModel written to {TFLITE_FILE}")
    print(f"Size: {len(tflite_model)} bytes")

    c_code = hex_to_c_array(tflite_model, "g_model")
    with C_MODEL_FILE.open("w", encoding="utf-8") as f:
        f.write(c_code)
    print("C++ model updated")


if __name__ == "__main__":
    quantize()
