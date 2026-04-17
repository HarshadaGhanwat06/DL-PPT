from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np
from scipy.signal import savgol_filter


@dataclass
class DatasetConfig:
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs/datasets")
    seed: int = 42
    pre_r_ms: float = 250.0
    post_r_ms: float = 500.0
    target_length: int = 160
    min_pep_ms: float = 10.0
    max_pep_ms: float = 250.0
    min_lvet_ms: float = 80.0
    max_lvet_ms: float = 400.0
    train_fraction: float = 0.65
    val_fraction: float = 0.18
    denoise_window_size: int = 5
    savgol_window_length: int = 11
    savgol_polyorder: int = 2
    target_variant: str = "base"
    avc_clip_min_ms: float = 120.0
    avc_clip_max_ms: float = 450.0


def _config_to_jsonable_dict(config: DatasetConfig) -> Dict[str, object]:
    raw = asdict(config)
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in raw.items()
    }


def _parse_subject_id(path: Path) -> str:
    return path.name.split("_")[0]

# Version 1 (incorrect)
# def _read_signal(group: h5py.Group, key: str) -> Tuple[np.ndarray, np.ndarray]:
#     signal_group = group[key]["value"]
#     data = np.asarray(signal_group["data"]["value"][()]).reshape(-1).astype(np.float64)
#     time = np.asarray(signal_group["time"]["value"][()]).reshape(-1).astype(np.float64)
#     return data, time


#Update v2
# def _read_signal(group, key):
#     signal_group = group[key]["value"]

#     # ✅ Correct access
#     data = np.asarray(signal_group["data"][()]).reshape(-1).astype(np.float64)
#     time = np.asarray(signal_group["time"][()]).reshape(-1).astype(np.float64)

#     return data, time

def _read_signal(group, key):
    signal_group = group[key]["value"]

    data = np.asarray(signal_group["data"]["value"][()]).reshape(-1).astype(np.float64)
    time = np.asarray(signal_group["time"]["value"][()]).reshape(-1).astype(np.float64)

    return data, time

def _moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return signal.copy()
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(signal, kernel, mode="same")


def moving_average(signal: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Apply a simple moving-average smoother to a 1D beat segment.

    This helper is used as a lightweight denoising fallback and as a
    pre-smoothing step before Savitzky-Golay filtering. Using a short window
    keeps the overall beat morphology intact while damping high-frequency noise.
    """
    if window_size <= 1:
        return signal.astype(np.float32, copy=True)
    kernel = np.ones(window_size, dtype=np.float64) / float(window_size)
    smoothed = np.convolve(signal.astype(np.float64), kernel, mode="same")
    return smoothed.astype(np.float32)


def denoise_segment(
    segment: np.ndarray,
    *,
    moving_average_window: int,
    savgol_window_length: int,
    savgol_polyorder: int,
) -> np.ndarray:
    """
    Denoise a resampled beat segment before normalization.

    The denoising is intentionally applied after resampling so every beat uses
    the same sample spacing, which makes the smoothing behavior consistent
    across the dataset. A short moving average removes small local spikes, and
    the Savitzky-Golay filter then smooths the signal while preserving peak and
    slope structure that is important for cardiac timing estimation.
    """
    smoothed = moving_average(segment, window_size=moving_average_window)

    # Savitzky-Golay requires an odd window length that is larger than the
    # polynomial order and not longer than the signal itself.
    window_length = min(savgol_window_length, int(smoothed.shape[0]))
    if window_length % 2 == 0:
        window_length -= 1
    minimum_window = savgol_polyorder + 2
    if minimum_window % 2 == 0:
        minimum_window += 1
    if window_length < minimum_window:
        return smoothed.astype(np.float32, copy=False)

    denoised = savgol_filter(
        smoothed.astype(np.float64),
        window_length=window_length,
        polyorder=savgol_polyorder,
    )
    return denoised.astype(np.float32)


def compute_dzdt(raw_icg: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    dt = float(np.median(np.diff(time_s)))
    dzdt = np.gradient(raw_icg, dt)
    baseline = _moving_average(dzdt, window=25)
    highpassed = dzdt - baseline
    smoothed = _moving_average(highpassed, window=5)
    return smoothed.astype(np.float32)


def load_record(path: Path) -> Dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        base = handle["measure"]["value"]
        ecg, time_s = _read_signal(base, "_030")
        raw_icg, _ = _read_signal(base, "_031")
        rpeaks, _ = _read_signal(base, "_032")
        avo, _ = _read_signal(base, "_033")
        pep_ms, _ = _read_signal(base, "_034")
        avc, _ = _read_signal(base, "_035")
        lvet_ms, _ = _read_signal(base, "_036")
    return {
        "path": np.array(str(path)),
        "subject_id": np.array(_parse_subject_id(path)),
        "time_s": time_s,
        "ecg": ecg.astype(np.float32),
        "raw_icg": raw_icg.astype(np.float32),
        "dzdt": compute_dzdt(raw_icg, time_s),
        "rpeaks_s": rpeaks,
        "avo_s": avo,
        "avc_s": avc,
        "pep_ms": pep_ms,
        "lvet_ms": lvet_ms,
    }


def _resample_segment(
    signal: np.ndarray,
    time_s: np.ndarray,
    start_s: float,
    end_s: float,
    target_length: int,
) -> np.ndarray | None:
    sample_points = np.linspace(start_s, end_s, target_length, dtype=np.float64)
    if start_s < time_s[0] or end_s > time_s[-1]:
        return None
    return np.interp(sample_points, time_s, signal).astype(np.float32)


def _heuristic_event_offsets_ms(segment: np.ndarray, relative_time_ms: np.ndarray) -> np.ndarray:
    avo_mask = (relative_time_ms >= 20.0) & (relative_time_ms <= 200.0)
    avc_mask = (relative_time_ms >= 120.0) & (relative_time_ms <= 450.0)
    avo_index = np.argmax(segment[avo_mask])
    avc_index = np.argmin(segment[avc_mask])
    avo_ms = float(relative_time_ms[avo_mask][avo_index])
    avc_ms = float(relative_time_ms[avc_mask][avc_index])
    return np.array([avo_ms, avc_ms], dtype=np.float32)


def _target_names_for_variant(target_variant: str) -> np.ndarray:
    """
    Return human-readable target names for the chosen dataset variant.

    Keeping the names variant-specific makes downstream scripts simpler because
    the saved metadata explicitly documents which label strategy was used.
    """
    return np.asarray(["pep_ms", "avc_ms"])


def _prepare_target(
    *,
    pep_value: float,
    avc_offset_ms: float,
    split_name: str,
    rng: np.random.Generator,
    config: DatasetConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the training target and the clean evaluation target for one beat.

    Only the clipped variant remains in this comparison branch. That keeps the
    target-noise experiment focused on the strategy that was actually useful.
    """
    if config.target_variant not in {"base", "clipped"}:
        raise ValueError(f"Unsupported target_variant={config.target_variant!r}")

    clipped_avc = float(np.clip(avc_offset_ms, config.avc_clip_min_ms, config.avc_clip_max_ms))
    reference_target = np.array([pep_value, clipped_avc], dtype=np.float32)

    if config.target_variant == "base":
        return np.array([pep_value, avc_offset_ms], dtype=np.float32), np.array([pep_value, avc_offset_ms], dtype=np.float32)

    if config.target_variant == "clipped":
        return reference_target.copy(), reference_target
    raise ValueError(f"Unsupported target_variant={config.target_variant!r}")


# def _iter_record_paths(data_dir: Path) -> Iterable[Path]:
#     return sorted(data_dir.rglob("*/*/*.h5"))
def _iter_record_paths(data_dir: Path) -> Iterable[Path]:
    paths = sorted(data_dir.rglob("*.h5"))
    print("DEBUG: Found files:", len(paths))
    for p in paths:
        print(" -", p)
    return paths

def inspect_dataset(config: DatasetConfig) -> Dict[str, object]:
    record_paths = list(_iter_record_paths(config.data_dir))
    beat_counts: List[int] = []
    pep_values: List[float] = []
    lvet_values: List[float] = []
    rr_values: List[float] = []
    subjects = set()
    invalid_beats = 0

    for path in record_paths:
        record = load_record(path)
        subjects.add(str(record["subject_id"]))
        rpeaks = record["rpeaks_s"]
        beat_counts.append(int(rpeaks.shape[0]))
        pep = (record["avo_s"] - rpeaks) * 1000.0
        lvet = (record["avc_s"] - record["avo_s"]) * 1000.0
        rr = np.diff(rpeaks) * 1000.0
        pep_values.extend(pep.tolist())
        lvet_values.extend(lvet.tolist())
        rr_values.extend(rr.tolist())
        invalid_beats += int(
            np.sum(
                (pep < config.min_pep_ms)
                | (pep > config.max_pep_ms)
                | (lvet < config.min_lvet_ms)
                | (lvet > config.max_lvet_ms)
            )
        )

    def summarize(values: List[float]) -> Dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        print("Beat counts:", beat_counts)
        print("Length:", len(beat_counts))
        if len(array) == 0:
            return {
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "count": 0
            }
        return {
            "min": float(np.min(array)),
            "p5": float(np.percentile(array, 5)),
            "mean": float(np.mean(array)),
            "p95": float(np.percentile(array, 95)),
            "max": float(np.max(array)),
        }

    summary = {
        "num_records": len(record_paths),
        "num_subjects": len(subjects),
        "subjects": sorted(subjects),
        "beats_per_record": summarize(beat_counts),
        "pep_ms": summarize(pep_values),
        "lvet_ms": summarize(lvet_values),
        "rr_ms": summarize(rr_values),
        "invalid_beats": invalid_beats,
        "config": _config_to_jsonable_dict(config),
    }
    return summary


def _split_subjects(subject_ids: List[str], config: DatasetConfig) -> Dict[str, List[str]]:
    rng = np.random.default_rng(config.seed)
    shuffled = list(subject_ids)
    rng.shuffle(shuffled)
    n_subjects = len(shuffled)
    n_train = max(1, int(round(n_subjects * config.train_fraction)))
    n_val = max(1, int(round(n_subjects * config.val_fraction)))
    n_train = min(n_train, n_subjects - 2)
    n_val = min(n_val, n_subjects - n_train - 1)
    train_subjects = sorted(shuffled[:n_train])
    val_subjects = sorted(shuffled[n_train : n_train + n_val])
    test_subjects = sorted(shuffled[n_train + n_val :])
    return {"train": train_subjects, "val": val_subjects, "test": test_subjects}


def build_dataset(config: DatasetConfig) -> Dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    pre_r_s = config.pre_r_ms / 1000.0
    post_r_s = config.post_r_ms / 1000.0
    relative_time_ms = np.linspace(-config.pre_r_ms, config.post_r_ms, config.target_length, dtype=np.float32)

    record_paths = list(_iter_record_paths(config.data_dir))
    subject_ids = sorted({_parse_subject_id(path) for path in record_paths})
    splits = _split_subjects(subject_ids, config)

    samples: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    reference_targets: List[np.ndarray] = []
    baseline_preds: List[np.ndarray] = []
    record_ids: List[str] = []
    sample_subject_ids: List[str] = []
    sample_split_names: List[str] = []
    quality_flags: List[int] = []

    for path in record_paths:
        record = load_record(path)
        subject_id = str(record["subject_id"])
        split_name = next(name for name, members in splits.items() if subject_id in members)
        rpeaks_s = record["rpeaks_s"]
        pep_ms = (record["avo_s"] - rpeaks_s) * 1000.0
        lvet_ms = (record["avc_s"] - record["avo_s"]) * 1000.0

        for beat_index, rpeak_s in enumerate(rpeaks_s):
            pep_value = float(pep_ms[beat_index])
            lvet_value = float(lvet_ms[beat_index])
            avc_offset_ms = pep_value + lvet_value
            is_valid = (
                config.min_pep_ms <= pep_value <= config.max_pep_ms
                and config.min_lvet_ms <= lvet_value <= config.max_lvet_ms
            )

            segment_dzdt = _resample_segment(
                signal=record["dzdt"],
                time_s=record["time_s"],
                start_s=float(rpeak_s - pre_r_s),
                end_s=float(rpeak_s + post_r_s),
                target_length=config.target_length,
            )
            segment_ecg = _resample_segment(
                signal=record["ecg"],
                time_s=record["time_s"],
                start_s=float(rpeak_s - pre_r_s),
                end_s=float(rpeak_s + post_r_s),
                target_length=config.target_length,
            )
            if segment_dzdt is None or segment_ecg is None:
                continue

            # Denoise each resampled beat before normalization so the model sees
            # cleaner waveform structure, especially in later regions that are
            # important for AVC prediction.
            segment_dzdt = denoise_segment(
                segment_dzdt,
                moving_average_window=config.denoise_window_size,
                savgol_window_length=config.savgol_window_length,
                savgol_polyorder=config.savgol_polyorder,
            )
            segment_ecg = denoise_segment(
                segment_ecg,
                moving_average_window=config.denoise_window_size,
                savgol_window_length=config.savgol_window_length,
                savgol_polyorder=config.savgol_polyorder,
            )

            # Normalize after denoising so each channel is centered and scaled
            # using the smoothed signal statistics rather than the noisier raw
            # segment statistics.
            segment_dzdt = (segment_dzdt - segment_dzdt.mean()) / (segment_dzdt.std() + 1e-6)
            segment_ecg = (segment_ecg - segment_ecg.mean()) / (segment_ecg.std() + 1e-6)
            multi_channel = np.stack([segment_dzdt, segment_ecg], axis=0).astype(np.float32)
            # Build versioned targets without altering the input signal. This
            # keeps the comparison focused entirely on noisy-label handling.
            target, reference_target = _prepare_target(
                pep_value=pep_value,
                avc_offset_ms=avc_offset_ms,
                split_name=split_name,
                rng=np.random.default_rng(config.seed),
                config=config,
            )
            baseline = _heuristic_event_offsets_ms(segment_dzdt, relative_time_ms)

            samples.append(multi_channel)
            targets.append(target)
            reference_targets.append(reference_target)
            baseline_preds.append(baseline)
            record_ids.append(f"{path.stem}::beat{beat_index:03d}")
            sample_subject_ids.append(subject_id)
            sample_split_names.append(split_name)
            quality_flags.append(int(is_valid))

    x = np.stack(samples).astype(np.float32)
    y = np.stack(targets).astype(np.float32)
    y_reference = np.stack(reference_targets).astype(np.float32)
    baseline = np.stack(baseline_preds).astype(np.float32)
    quality = np.asarray(quality_flags, dtype=np.int8)
    record_id_array = np.asarray(record_ids)
    subject_array = np.asarray(sample_subject_ids)
    split_array = np.asarray(sample_split_names)

    summary: Dict[str, object] = {
        "config": _config_to_jsonable_dict(config),
        "subject_splits": splits,
        "target_variant": config.target_variant,
        "total_segments": int(x.shape[0]),
        "valid_segments": int(quality.sum()),
        "invalid_segments": int((1 - quality).sum()),
    }

    np.savez_compressed(
        config.output_dir / "all_segments.npz",
        x=x,
        y=y,
        y_reference=y_reference,
        baseline=baseline,
        quality=quality,
        record_id=record_id_array,
        subject_id=subject_array,
        split=split_array,
        relative_time_ms=relative_time_ms,
        target_names=_target_names_for_variant(config.target_variant),
        reference_target_names=np.asarray(["pep_ms", "avc_ms"]),
        channel_names=np.asarray(["dzdt", "ecg"]),
    )

    for split_name in ("train", "val", "test"):
        mask = (split_array == split_name) & (quality == 1)
        np.savez_compressed(
            config.output_dir / f"{split_name}.npz",
            x=x[mask],
            y=y[mask],
            y_reference=y_reference[mask],
            baseline=baseline[mask],
            quality=quality[mask],
            record_id=record_id_array[mask],
            subject_id=subject_array[mask],
            split=split_array[mask],
            relative_time_ms=relative_time_ms,
            target_names=_target_names_for_variant(config.target_variant),
            reference_target_names=np.asarray(["pep_ms", "avc_ms"]),
            channel_names=np.asarray(["dzdt", "ecg"]),
        )
        summary[f"{split_name}_segments"] = int(mask.sum())

    with (config.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary
