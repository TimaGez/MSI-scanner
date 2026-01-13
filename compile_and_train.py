import os
import argparse
import json
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import tensorflow as tf


# -------------------------
# Dataset structure (yours)
# -------------------------

@dataclass
class Sample:
    session_dir: str
    label: int
    ml_tensor_path: Optional[str] = None
    corr660_path: Optional[str] = None
    corr730_path: Optional[str] = None
    raw660_path: Optional[str] = None
    raw730_path: Optional[str] = None


def find_first_matching_file(folder: str, suffix: str) -> Optional[str]:
    for fn in os.listdir(folder):
        if fn.endswith(suffix):
            return os.path.join(folder, fn)
    return None


def scan_session_folder(session_dir: str) -> Sample:
    ml = find_first_matching_file(session_dir, "-ML-tensor.npy")
    corr660 = find_first_matching_file(session_dir, "-660nm-corr.npy")
    corr730 = find_first_matching_file(session_dir, "-730nm-corr.npy")
    raw660  = find_first_matching_file(session_dir, "-660nm-raw.npy")
    raw730  = find_first_matching_file(session_dir, "-730nm-raw.npy")

    return Sample(
        session_dir=session_dir,
        label=-1,
        ml_tensor_path=ml,
        corr660_path=corr660,
        corr730_path=corr730,
        raw660_path=raw660,
        raw730_path=raw730,
    )


def find_session_dirs(data_root: str) -> List[str]:
    session_dirs = []
    for root, _, files in os.walk(data_root):
        has_ml = any(f.endswith("-ML-tensor.npy") for f in files)
        has_four = (
            any(f.endswith("-660nm-corr.npy") for f in files) and
            any(f.endswith("-730nm-corr.npy") for f in files) and
            any(f.endswith("-660nm-raw.npy")  for f in files) and
            any(f.endswith("-730nm-raw.npy")  for f in files)
        )
        if has_ml or has_four:
            session_dirs.append(root)
    return sorted(session_dirs)


def build_class_map(class_names_csv: str) -> Dict[str, int]:
    names = [c.strip() for c in class_names_csv.split(",") if c.strip()]
    return {name: i for i, name in enumerate(names)}


def infer_label_from_path(session_dir: str, class_map: Dict[str, int]) -> Optional[int]:
    parts = [p.lower() for p in session_dir.split(os.sep)]
    for cls, idx in class_map.items():
        if cls.lower() in parts:
            return idx
    return None


def load_labels_json(path: str) -> Dict[str, int]:
    with open(path, "r") as f:
        m = json.load(f)
    return {str(k): int(v) for k, v in m.items()}


# -------------------------
# Tensor loading / shaping
# -------------------------

def load_npy(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        obj = arr.item()
        if isinstance(obj, dict):
            arr = np.array(list(obj.values())[0])
        elif isinstance(obj, (tuple, list)) and len(obj) > 0:
            arr = np.array(obj[0])
        else:
            arr = np.array(obj)
    return np.array(arr)


def to_hwc(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)

    if x.ndim == 2:
        return x[..., None]

    if x.ndim == 3:
        # (H,W,C)
        if x.shape[-1] in (1, 2, 3, 4):
            return x
        # (C,H,W)
        if x.shape[0] in (1, 2, 3, 4) and x.shape[1] > 8 and x.shape[2] > 8:
            return np.transpose(x, (1, 2, 0))
        # (1,H,W)
        if x.shape[0] == 1 and x.shape[1] > 8 and x.shape[2] > 8:
            return x[0][..., None]

    raise ValueError(f"Unsupported tensor shape: {x.shape}")


def resize_and_norm(x: np.ndarray, img_size: int, normalize: str) -> np.ndarray:
    x = x.astype(np.float32)
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    x_tf = tf.image.resize(x_tf, (img_size, img_size), method="bilinear")
    x = x_tf.numpy()

    if normalize == "minmax":
        mn, mx = float(x.min()), float(x.max())
        x = (x - mn) / (mx - mn + 1e-6)

    return x.astype(np.float32)


def build_input_tensor(sample: Sample, img_size: int, normalize: str, mode: str) -> np.ndarray:
    if mode == "ml":
        if not sample.ml_tensor_path:
            raise ValueError(f"Missing ML-tensor in {sample.session_dir}")
        x = to_hwc(load_npy(sample.ml_tensor_path))

    elif mode == "corr":
        if not (sample.corr660_path and sample.corr730_path):
            raise ValueError(f"Missing corrected tensors in {sample.session_dir}")
        c660 = to_hwc(load_npy(sample.corr660_path))
        c730 = to_hwc(load_npy(sample.corr730_path))
        x = np.concatenate([c660, c730], axis=-1)

    elif mode == "raw":
        if not (sample.raw660_path and sample.raw730_path):
            raise ValueError(f"Missing raw tensors in {sample.session_dir}")
        r660 = to_hwc(load_npy(sample.raw660_path))
        r730 = to_hwc(load_npy(sample.raw730_path))
        x = np.concatenate([r660, r730], axis=-1)

    elif mode == "corr_raw":
        if not (sample.corr660_path and sample.corr730_path and sample.raw660_path and sample.raw730_path):
            raise ValueError(f"Missing corr/raw tensors in {sample.session_dir}")
        c660 = to_hwc(load_npy(sample.corr660_path))
        c730 = to_hwc(load_npy(sample.corr730_path))
        r660 = to_hwc(load_npy(sample.raw660_path))
        r730 = to_hwc(load_npy(sample.raw730_path))
        x = np.concatenate([c660, c730, r660, r730], axis=-1)

    else:
        raise ValueError("mode must be one of: ml, corr, raw, corr_raw")

    x = resize_and_norm(x, img_size, normalize)
    return x


# -------------------------
# Augmentation (train-time + robustness test)
# -------------------------

def augment_tensor(x: tf.Tensor) -> tf.Tensor:
    # keep augmentations "physically plausible"
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)

    # intensity jitter per channel
    jitter = tf.random.uniform([1, 1, tf.shape(x)[-1]], 0.95, 1.05)
    x = tf.clip_by_value(x * jitter, 0.0, 1.0)
    return x


# -------------------------
# Model (EdgeTPU-friendly CNN)
# -------------------------

def build_edgetpu_cnn(img_size: int, in_channels: int, num_classes: int) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(img_size, img_size, in_channels))

    x = tf.keras.layers.Conv2D(24, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.DepthwiseConv2D(3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(32, 1, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.DepthwiseConv2D(3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(48, 1, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.DepthwiseConv2D(3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, 1, padding="same", activation="relu")(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(num_classes)(x)  # logits
    return tf.keras.Model(inp, out, name="edgetpu_cnn")


# -------------------------
# Dataset builders
# -------------------------

def make_train_dataset(samples: List[Sample], img_size: int, normalize: str, mode: str,
                       batch_size: int, seed: int) -> tf.data.Dataset:
    # generator yields x,y
    x0 = build_input_tensor(samples[0], img_size, normalize, mode)
    out_sig = (
        tf.TensorSpec(shape=x0.shape, dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    def gen():
        for s in samples:
            x = build_input_tensor(s, img_size, normalize, mode)
            y = np.int32(s.label)
            yield x, y

    ds = tf.data.Dataset.from_generator(gen, output_signature=out_sig)
    ds = ds.shuffle(min(len(samples), 2048), seed=seed, reshuffle_each_iteration=True)

    def map_aug(x, y):
        x = augment_tensor(x)
        return x, y

    ds = ds.map(map_aug, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def make_eval_tensor(sample: Sample, img_size: int, normalize: str, mode: str) -> np.ndarray:
    return build_input_tensor(sample, img_size, normalize, mode)


# -------------------------
# Metrics (tiny-N friendly)
# -------------------------

def softmax_prob(logits: np.ndarray, positive_class: int = 1) -> float:
    # logits shape: (num_classes,)
    ex = np.exp(logits - np.max(logits))
    p = ex / (np.sum(ex) + 1e-9)
    return float(p[positive_class])


def confusion_counts(y_true: List[int], y_pred: List[int], pos_label: int = 1) -> Dict[str, int]:
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == pos_label and yp == pos_label)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt != pos_label and yp != pos_label)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != pos_label and yp == pos_label)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == pos_label and yp != pos_label)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def balanced_accuracy(cm: Dict[str, int]) -> float:
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    tpr = tp / (tp + fn + 1e-9)
    tnr = tn / (tn + fp + 1e-9)
    return float(0.5 * (tpr + tnr))


def auc_approx(y_true: List[int], y_score: List[float], pos_label: int = 1) -> float:
    """
    Simple AUC computation (Mann-Whitney U / ranking based).
    With N=5 this will be coarse, but still usable as a "signal exists" metric.
    """
    pos = [(s, yt) for s, yt in zip(y_score, y_true) if yt == pos_label]
    neg = [(s, yt) for s, yt in zip(y_score, y_true) if yt != pos_label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    wins = 0.0
    total = 0.0
    for ps, _ in pos:
        for ns, _ in neg:
            total += 1
            if ps > ns:
                wins += 1
            elif ps == ns:
                wins += 0.5
    return float(wins / (total + 1e-9))


# -------------------------
# Export INT8 TFLite
# -------------------------

def export_int8_tflite(model: tf.keras.Model, rep_ds: tf.data.Dataset, out_path: str):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def rep_gen():
        for x, _ in rep_ds.take(200):
            yield [tf.cast(x, tf.float32)]
    converter.representative_dataset = rep_gen

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)


# -------------------------
# Main evaluation loop
# -------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_csv(path: str, header: List[str], rows: List[List]):
    with open(path, "w", newline="") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", required=True)
    ap.add_argument("--class_names", default="negative,positive", help="Comma-separated class names; positive should be index 1.")
    ap.add_argument("--labels_json", default=None, help="Optional JSON mapping session folder name -> label id.")
    ap.add_argument("--mode", default="ml", choices=["ml", "corr", "raw", "corr_raw"])

    ap.add_argument("--img_size", type=int, default=96)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)

    ap.add_argument("--normalize", default="minmax", choices=["minmax", "none"])
    ap.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated seeds to repeat each LOSO fold.")
    ap.add_argument("--robust_aug_n", type=int, default=50, help="For each held-out scan, run N augmentations to estimate prediction stability.")
    ap.add_argument("--threshold", type=float, default=0.5, help="Decision threshold on mean probability.")
    ap.add_argument("--out_dir", default="tinyN_results")
    ap.add_argument("--export_tflite", action="store_true", help="Export an INT8 TFLite from the final trained model on all data.")

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    class_map = {c.strip(): i for i, c in enumerate(args.class_names.split(","))}
    if len(class_map) < 2:
        raise RuntimeError("Need at least 2 classes.")
    pos_label = 1  # assume index 1 is positive like you asked

    label_map = load_labels_json(args.labels_json) if args.labels_json else None

    session_dirs = find_session_dirs(args.data_root)
    if not session_dirs:
        raise RuntimeError("No session folders found.")

    samples: List[Sample] = []
    for sd in session_dirs:
        s = scan_session_folder(sd)

        # label
        if label_map is not None:
            folder_name = os.path.basename(sd)
            if folder_name not in label_map:
                continue
            s.label = label_map[folder_name]
        else:
            lab = infer_label_from_path(sd, class_map)
            if lab is None:
                continue
            s.label = lab

        # require tensors for chosen mode
        if args.mode == "ml" and not s.ml_tensor_path:
            continue
        if args.mode == "corr" and not (s.corr660_path and s.corr730_path):
            continue
        if args.mode == "raw" and not (s.raw660_path and s.raw730_path):
            continue
        if args.mode == "corr_raw" and not (s.corr660_path and s.corr730_path and s.raw660_path and s.raw730_path):
            continue

        samples.append(s)

    if len(samples) < 5:
        print(f"[WARN] Only {len(samples)} samples found. Results will be extremely noisy.")
    print(f"[INFO] Using {len(samples)} labeled session samples for LOSO evaluation.")

    # Prepare seeds
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    # Infer input channels once
    x0 = build_input_tensor(samples[0], args.img_size, args.normalize, args.mode)
    in_channels = x0.shape[-1]
    num_classes = len(class_map)

    # Storage
    per_sample_rows = []  # session, true, mean_p, std_p, pred, fold_idx
    loss_curve_rows = []  # seed, fold_idx, epoch, loss, val_loss (val is train held-out? we won't use val)

    # LOSO folds
    for fold_idx in range(len(samples)):
        test_sample = samples[fold_idx]
        train_samples = [s for i, s in enumerate(samples) if i != fold_idx]

        # Collect predictions across seeds
        seed_probs = []
        seed_losses = []

        x_test_np = make_eval_tensor(test_sample, args.img_size, args.normalize, args.mode)
        x_test = tf.convert_to_tensor(x_test_np[None, ...], dtype=tf.float32)

        for seed in seeds:
            tf.keras.utils.set_random_seed(seed)

            model = build_edgetpu_cnn(args.img_size, in_channels, num_classes)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-3),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            )

            train_ds = make_train_dataset(train_samples, args.img_size, args.normalize, args.mode, args.batch_size, seed=seed)

            hist = model.fit(train_ds, epochs=args.epochs, verbose=0)
            losses = hist.history.get("loss", [])
            # record loss curve
            for ep, l in enumerate(losses, start=1):
                loss_curve_rows.append([seed, fold_idx, ep, float(l)])

            # Predict on held-out scan (no augmentation)
            logits = model(x_test, training=False).numpy()[0]
            p_pos = softmax_prob(logits, positive_class=pos_label)
            seed_probs.append(p_pos)

            # Robustness test: N augmented variants of same scan
            aug_probs = []
            for _ in range(args.robust_aug_n):
                x_aug = augment_tensor(x_test[0]).numpy()[None, ...]
                logits_aug = model(tf.convert_to_tensor(x_aug, dtype=tf.float32), training=False).numpy()[0]
                aug_probs.append(softmax_prob(logits_aug, positive_class=pos_label))
            # store robustness as extra stats per seed (mean/std across augs)
            seed_losses.append((float(np.mean(aug_probs)), float(np.std(aug_probs))))

        mean_p = float(np.mean(seed_probs))
        std_p = float(np.std(seed_probs))
        pred = 1 if mean_p >= args.threshold else 0

        # robustness summarized across seeds: mean of means / mean of stds
        rob_mean = float(np.mean([m for (m, s) in seed_losses]))
        rob_std  = float(np.mean([s for (m, s) in seed_losses]))

        per_sample_rows.append([
            os.path.basename(test_sample.session_dir),
            test_sample.label,
            mean_p,
            std_p,
            pred,
            fold_idx,
            rob_mean,
            rob_std,
        ])

        print(f"[FOLD {fold_idx}] {os.path.basename(test_sample.session_dir)} true={test_sample.label} "
              f"p_pos={mean_p:.3f}±{std_p:.3f} pred={pred} | aug_stability_mean={rob_mean:.3f} avg_aug_std={rob_std:.3f}")

    # Compute final summary metrics across folds
    y_true = [r[1] for r in per_sample_rows]
    y_pred = [r[4] for r in per_sample_rows]
    y_score = [r[2] for r in per_sample_rows]

    cm = confusion_counts(y_true, y_pred, pos_label=pos_label)
    bacc = balanced_accuracy(cm)
    acc = float(np.mean([1 if yt == yp else 0 for yt, yp in zip(y_true, y_pred)]))
    auc = auc_approx(y_true, y_score, pos_label=pos_label)

    # Save artifacts
    save_csv(
        os.path.join(args.out_dir, "predictions.csv"),
        ["session", "y_true", "p_pos_mean", "p_pos_std_across_seeds", "y_pred", "fold",
         "p_pos_aug_mean", "p_pos_aug_std_mean"],
        per_sample_rows,
    )

    save_csv(
        os.path.join(args.out_dir, "loss_curves.csv"),
        ["seed", "fold", "epoch", "train_loss"],
        loss_curve_rows,
    )

    save_csv(
        os.path.join(args.out_dir, "confusion_matrix.csv"),
        ["tp", "tn", "fp", "fn"],
        [[cm["tp"], cm["tn"], cm["fp"], cm["fn"]]],
    )

    summary = {
        "n_samples": len(samples),
        "seeds": seeds,
        "mode": args.mode,
        "img_size": args.img_size,
        "epochs": args.epochs,
        "threshold": args.threshold,
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "auc_approx": auc,
        "confusion": cm,
        "notes": "Tiny-N LOSO evaluation. Metrics are HIGH VARIANCE; treat as feasibility only.",
    }

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write("Tiny-N LOSO Evaluation Summary\n")
        f.write(json.dumps(summary, indent=2))
        f.write("\n")

    print("\n[SUMMARY]")
    print(json.dumps(summary, indent=2))

    # Optional: train on ALL data and export TFLite (for deployment only, not evaluation)
    if args.export_tflite:
        print("\n[EXPORT] Training on ALL data for deployment export...")
        tf.keras.utils.set_random_seed(0)

        model = build_edgetpu_cnn(args.img_size, in_channels, num_classes)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )

        all_ds = make_train_dataset(samples, args.img_size, args.normalize, args.mode, args.batch_size, seed=0)
        model.fit(all_ds, epochs=args.epochs, verbose=0)

        tflite_path = os.path.join(args.out_dir, "deploy_int8.tflite")
        export_int8_tflite(model, all_ds, tflite_path)
        print(f"[OK] Wrote {tflite_path}")
        print("Next on Pi: edgetpu_compiler deploy_int8.tflite")

if __name__ == "__main__":
    main()
