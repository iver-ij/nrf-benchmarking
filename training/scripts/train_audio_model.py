import random
import warnings
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from scipy.io import wavfile
from dataset_paths import resolve_dataset_dir

DATASET_PATH = resolve_dataset_dir()
TARGET_WORD = "on"

SAMPLE_RATE = 16000
NUM_SAMPLES = 16000
FFT_SIZE = 512
FRAME_STEP = 320
NUM_MEL_BINS = 40
INPUT_SHAPE = (49, 40, 1)
BATCH_SIZE = 64
EPOCHS = 30
SEED = 42
EPSILON = 1e-6
DSP_BACKEND = os.getenv("DSP_BACKEND", "cmsis").lower()

# Minimal, proven balancing knobs for the 3-class spectral classifier
UNKNOWN_TO_TARGET_RATIO = 2.0
SILENCE_TO_TARGET_RATIO = 1.0

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from cmsis_host_dsp import extract_logmel_cmsis
except Exception:
    extract_logmel_cmsis = None


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_audio(item):
    if isinstance(item, (Path, str)):
        try:
            _, data = wavfile.read(item)
        except Exception:
            return np.zeros(NUM_SAMPLES, dtype=np.float32)
    else:
        data = item

    if data.dtype != np.float32:
        data = data.astype(np.float32) / 32768.0

    if data.ndim > 1:
        data = data[:, 0]

    if len(data) < NUM_SAMPLES:
        padding = NUM_SAMPLES - len(data)
        data = np.concatenate([data, np.zeros(padding, dtype=np.float32)])
    else:
        data = data[:NUM_SAMPLES]
    return data


def _extract_features_tf(audio_array):
    audio_tensor = tf.convert_to_tensor(audio_array, dtype=tf.float32)
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
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=NUM_MEL_BINS,
        num_spectrogram_bins=stft.shape[-1],
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=20.0,
        upper_edge_hertz=4000.0,
    )

    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrogram.set_shape(
        spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:])
    )
    log_mel_spectrogram = tf.math.log(mel_spectrogram + EPSILON)
    return tf.expand_dims(log_mel_spectrogram, axis=-1)


def extract_features(audio_array):
    if DSP_BACKEND == "cmsis":
        if extract_logmel_cmsis is None:
            raise RuntimeError("DSP_BACKEND=cmsis requested but cmsis_host_dsp is unavailable")
        return extract_logmel_cmsis(np.asarray(audio_array, dtype=np.float32), add_channel=True)
    return _extract_features_tf(audio_array)


def augment_audio(audio: np.ndarray) -> np.ndarray:
    # Time shift
    shift = random.randint(-1000, 1000)
    audio = np.roll(audio, shift)

    # Small gain jitter
    gain = random.uniform(0.8, 1.2)
    audio = np.clip(audio * gain, -1.0, 1.0)

    # Small additive noise
    if random.random() < 0.3:
        audio = np.clip(audio + np.random.normal(0.0, 0.003, size=audio.shape), -1.0, 1.0)

    return audio.astype(np.float32)


def create_dataset(items, labels, is_training=True):
    def generator():
        for i in range(len(items)):
            audio = load_audio(items[i])
            if is_training and random.random() < 0.5:
                audio = augment_audio(audio)
            yield extract_features(audio), labels[i]

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=INPUT_SHAPE, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )
    if is_training:
        dataset = dataset.shuffle(buffer_size=min(2000, len(items)), seed=SEED)

    return dataset.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)


def read_split_file(filename: str) -> set[str]:
    p = DATASET_PATH / filename
    if not p.exists():
        return set()
    return {line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()}


def split_paths(paths: list[Path], val_list: set[str], test_list: set[str]) -> tuple[list[Path], list[Path]]:
    train, val = [], []
    for p in paths:
        rel = p.relative_to(DATASET_PATH).as_posix()
        if rel in test_list:
            continue
        if rel in val_list:
            val.append(p)
        else:
            train.append(p)
    return train, val


def build_background_chunks() -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = random.Random(SEED)
    chunks = []
    for file in (DATASET_PATH / "_background_noise_").glob("*.wav"):
        try:
            _, audio = wavfile.read(file)
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32) / 32768.0
            if audio.ndim > 1:
                audio = audio[:, 0]
            for i in range(0, max(1, len(audio) - NUM_SAMPLES + 1), NUM_SAMPLES):
                clip = audio[i : i + NUM_SAMPLES]
                if len(clip) < NUM_SAMPLES:
                    clip = np.pad(clip, (0, NUM_SAMPLES - len(clip)), mode="constant")
                chunks.append(clip.astype(np.float32))
        except Exception:
            continue

    rng.shuffle(chunks)
    split_idx = int(0.8 * len(chunks))
    return chunks[:split_idx], chunks[split_idx:]


def cap_items(items: list, cap: int) -> list:
    if cap <= 0 or not items:
        return []
    if len(items) <= cap:
        return list(items)
    return random.sample(items, cap)


def build_class_weights(labels: list[int]) -> dict[int, float]:
    counts = {0: 0, 1: 0, 2: 0}
    for y in labels:
        counts[y] += 1
    non_zero = [c for c in counts.values() if c > 0]
    median = float(np.median(non_zero)) if non_zero else 1.0
    return {k: (median / v if v > 0 else 1.0) for k, v in counts.items()}


if __name__ == "__main__":
    set_reproducible_seed(SEED)

    print(f"TRAINING START: TARGET '{TARGET_WORD}' = LABEL 2")

    val_list = read_split_file("validation_list.txt")
    test_list = read_split_file("testing_list.txt")

    # Target word
    target_all = list((DATASET_PATH / TARGET_WORD).glob("*.wav"))
    target_train, target_val = split_paths(target_all, val_list, test_list)

    # Unknown words
    unknown_all = []
    for folder in [d for d in DATASET_PATH.iterdir() if d.is_dir()]:
        if folder.name in {TARGET_WORD, "_background_noise_"}:
            continue
        unknown_all.extend(folder.glob("*.wav"))
    unknown_train, unknown_val = split_paths(unknown_all, val_list, test_list)

    # Silence/background
    bg_train, bg_val = build_background_chunks()

    # Minimal proven balancing for better target recall
    target_train_n = len(target_train)
    target_val_n = len(target_val)

    unknown_train = cap_items(unknown_train, int(target_train_n * UNKNOWN_TO_TARGET_RATIO))
    bg_train = cap_items(bg_train, int(target_train_n * SILENCE_TO_TARGET_RATIO))

    unknown_val = cap_items(unknown_val, int(target_val_n * UNKNOWN_TO_TARGET_RATIO))
    bg_val = cap_items(bg_val, int(target_val_n * SILENCE_TO_TARGET_RATIO))

    print(
        "Train class sizes | "
        f"silence={len(bg_train)} unknown={len(unknown_train)} target={len(target_train)}"
    )
    print(
        "Val class sizes   | "
        f"silence={len(bg_val)} unknown={len(unknown_val)} target={len(target_val)}"
    )

    train_items = []
    train_labels = []
    train_items.extend(bg_train)
    train_labels.extend([0] * len(bg_train))
    train_items.extend(unknown_train)
    train_labels.extend([1] * len(unknown_train))
    train_items.extend(target_train)
    train_labels.extend([2] * len(target_train))

    val_items = []
    val_labels = []
    val_items.extend(bg_val)
    val_labels.extend([0] * len(bg_val))
    val_items.extend(unknown_val)
    val_labels.extend([1] * len(unknown_val))
    val_items.extend(target_val)
    val_labels.extend([2] * len(target_val))

    # Shuffle once, deterministically
    train_perm = np.random.permutation(len(train_items))
    train_items = [train_items[i] for i in train_perm]
    train_labels = [train_labels[i] for i in train_perm]

    val_perm = np.random.permutation(len(val_items))
    val_items = [val_items[i] for i in val_perm]
    val_labels = [val_labels[i] for i in val_perm]

    train_ds = create_dataset(train_items, train_labels, is_training=True)
    val_ds = create_dataset(val_items, val_labels, is_training=False)

    class_weights = build_class_weights(train_labels)
    print(f"Class weights: {class_weights}")

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=INPUT_SHAPE),
            tf.keras.layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.DepthwiseConv2D((3, 3), padding="same", activation="relu"),
            tf.keras.layers.Conv2D(32, (1, 1), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.DepthwiseConv2D((3, 3), padding="same", activation="relu"),
            tf.keras.layers.Conv2D(64, (1, 1), activation="relu"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    model.save("training/outputs/model.keras")
    print("Model saved: training/outputs/model.keras")
