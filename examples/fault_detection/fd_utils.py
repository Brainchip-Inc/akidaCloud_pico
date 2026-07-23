"""
Utilities for the bearing fault detection demo notebook: model definition, data loading,
streaming inference helpers, evaluation metrics, and plotting. Keeps the notebook focused
on the main pipeline.

Dependencies: numpy, h5py, json, textwrap (stdlib/common), tensorflow, tf_keras,
              akida_models, sklearn — all available in any Akida environment.
              No external repo dependency.
"""

import json
import os
import textwrap

import h5py
import numpy as np

# UORED-VAFCLS fault label order (multi-label: each is independent)
FAULT_NAMES = ['inner', 'outer', 'ball', 'cage']
FAULT_COLORS = ['#0072B2', '#009E73', '#CC79A7', '#E69F00']  # colorblind-safe (Okabe-Ito): blue, green, pink, amber
SAMPLE_RATE = 42000  # UORED-VAFCLS is 42 kHz

# Health state mapping
HEALTH_STATES = ['healthy', 'fault_1', 'fault_2']

# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

def tenn_recurrent_uored(input_shape=(4096, 1), num_classes=4, input_scaling=(2**15, 0)):
    """Instantiates a TENN recurrent architecture for UORED-VAFCLS bearing fault classification.

    Multi-label model: outputs raw logits (no activation) for 4 fault types
    (inner race, outer race, ball, cage). Use with BinaryCrossentropy(from_logits=True).

    Args:
        input_shape (tuple, optional): the input shape. Defaults to (4096, 1).
        num_classes (int, optional): number of fault classes. Defaults to 4.
        input_scaling (tuple, optional): scale factor and offset for int16 input rescaling.
            Following Akida convention, the scale factor is used as a divisor.
            Defaults to (2**15, 0).

    Returns:
        keras.Model: a TENN recurrent model for UORED-VAFCLS
    """
    import tensorflow as tf
    from tf_keras.models import Model
    from tf_keras.layers import Input, GlobalAveragePooling1D, SpatialDropout1D, Rescaling
    from akida_models.layer_blocks import kernelized_block

    num_coeffs = 32
    channels = [8, 16, 32, 32, 64]
    subsampling_pattern = [8, 4, 2, 2, 2]   # total 256x -> 4096/256 = 16 final temporal steps

    inputs = Input(shape=input_shape, dtype=tf.int16, name="input")

    scale, offset = input_scaling
    x = Rescaling(1. / scale, offset, name="rescaling")(inputs)

    for i, (channel, subsample) in enumerate(zip(channels, subsampling_pattern)):
        x = kernelized_block(x, num_coeffs, channel, subsampling=subsample,
                             add_batchnorm=True, relu_activation='ReLU', name=f'ssm_layer_{i}')
        x = SpatialDropout1D(0.1)(x)

    x = kernelized_block(x, num_coeffs, num_classes, subsampling=False, add_batchnorm=False,
                         relu_activation=False, name='ssm_layer_head')
    x = GlobalAveragePooling1D(name='gap')(x)

    return Model(inputs, x, name="tenn_recurrent_uored")


# ---------------------------------------------------------------------------
# Split generation
# ---------------------------------------------------------------------------

def generate_default_split(output_path):
    """Generate a default bearing-wise train/test split JSON.

    Test set: one bearing per fault type (1=inner, 6=outer, 11=ball, 16=cage).
    Train set: all remaining bearings (2-5, 7-10, 12-15, 17-20).

    Args:
        output_path (str): Path to write the split JSON file.

    Returns:
        dict: The split dictionary that was written.
    """
    split = {
        "split_id": 0,
        "test_bearings": [1, 6, 11, 16],
        "train_bearings": [2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20],
        "description": ("Default split: one test bearing per fault type "
                        "(inner=1, outer=6, ball=11, cage=16).")
    }
    with open(output_path, 'w') as f:
        json.dump(split, f, indent=4)
    print(f"Generated default split at: {output_path}")
    return split


# ---------------------------------------------------------------------------
# Data loading  (pure h5py + json — no external repo needed)
# ---------------------------------------------------------------------------

def _load_hdf5_split(hdf5_path, bearing_ids, segment_points, stride=None):
    """Load and segment signals for a list of bearing IDs from the HDF5 file.

    Preprocessing matches uored_train.py:
      - Channel 0 (accelerometer) only
      - Per-recording peak normalization to [-1, 1]
      - Scale to int16 range

    Returns:
        signals  (list of np.ndarray): each (segment_points, 1) int16
        labels   (list of np.ndarray): each (4,) float32 multi-label
        metadata (list of dict): bearing_id, health_state, fault_type, segment_index
    """
    if stride is None:
        stride = segment_points

    signals, labels, metadata = [], [], []

    with h5py.File(hdf5_path, 'r') as hf:
        bearings_group = hf['bearings']
        for bid in bearing_ids:
            # Split JSON bearing IDs are 1-indexed; HDF5 keys are 0-indexed
            # ('bearing_00' == bearing ID 1), matching akida_models/uored_train.py.
            # Using the ID directly would load the wrong (mostly training) bearings
            # and leak them into the test set.
            bname = f'bearing_{bid - 1:02d}'
            if bname not in bearings_group:
                print(f"  [WARNING] {bname} not found in HDF5 — skipping.")
                continue
            bearing = bearings_group[bname]
            fault_type = bearing.attrs.get('fault_type', 'unknown')

            for state in HEALTH_STATES:
                if state not in bearing:
                    continue
                ds = bearing[state]
                sig = ds[:]                                       # (420000, n_ch)
                lbl = np.array(ds.attrs['multi_label'], dtype=np.float32)

                # Use channel 0 (accelerometer) only
                sig_ch0 = sig[:, 0].astype(np.float32)           # (420000,)

                # Per-recording peak normalization to [-1, 1]
                peak = np.max(np.abs(sig_ch0))
                if peak > 0:
                    sig_ch0 = sig_ch0 / peak

                # Scale to int16
                sig_int16 = (sig_ch0 * (2**15 - 1)).astype(np.int16)

                n = sig_int16.shape[0]
                seg_idx = 0
                for start in range(0, n - segment_points + 1, stride):
                    segment = sig_int16[start:start + segment_points]
                    signals.append(segment[:, np.newaxis])        # (segment_points, 1)
                    labels.append(lbl)
                    metadata.append({
                        'bearing_id': bid,
                        'health_state': state,
                        'fault_type': fault_type,
                        'segment_index': seg_idx,
                        'segment_start': start,
                    })
                    seg_idx += 1

    return signals, labels, metadata


def load_test_data(hdf5_path, split_id=0, segment_points=4096):
    """Load UORED-VAFCLS test data for a given split.

    Looks for the split JSON in the same directory as the HDF5 file.

    Args:
        hdf5_path (str): Path to uored_full.h5
        split_id (int): Split index (default 0).
        segment_points (int): Segment length in samples. Defaults to 4096.

    Returns:
        X_test (np.ndarray): shape (n_samples, segment_points, 1) int16
        y_test (np.ndarray): shape (n_samples, 4) float32 multi-label
        metadata (list): list of dicts per sample
    """
    data_dir = os.path.dirname(hdf5_path)
    split_file = os.path.join(data_dir, f'split_{split_id:03d}.json')

    # Auto-generate default split if missing
    if not os.path.exists(split_file) and split_id == 0:
        generate_default_split(split_file)

    with open(split_file) as f:
        split = json.load(f)
    test_bearings = split['test_bearings']

    signals, labels, metadata = _load_hdf5_split(
        hdf5_path, test_bearings, segment_points, stride=segment_points
    )

    print(f"Loaded test data — split {split_id}")
    print(f"  Bearings : {test_bearings}")
    print(f"  Segments : {len(signals)}")
    if signals:
        print(f"  Shape    : {signals[0].shape}, dtype: {signals[0].dtype}")

    return np.array(signals), np.array(labels), metadata


def load_calibration_data(hdf5_path, split_id=0, segment_points=4096, n_samples=256):
    """Load a subset of training data to use as calibration data for quantization.

    Looks for the split JSON in the same directory as the HDF5 file.

    Args:
        hdf5_path (str): Path to uored_full.h5
        split_id (int): Split index (default 0).
        segment_points (int): Segment length in samples.
        n_samples (int): Number of calibration samples to return.

    Returns:
        np.ndarray: shape (n_samples, segment_points, 1) int16
    """
    data_dir = os.path.dirname(hdf5_path)
    split_file = os.path.join(data_dir, f'split_{split_id:03d}.json')

    if not os.path.exists(split_file) and split_id == 0:
        generate_default_split(split_file)

    with open(split_file) as f:
        split = json.load(f)
    train_bearings = split['train_bearings']

    signals, _, _ = _load_hdf5_split(
        hdf5_path, train_bearings, segment_points, stride=segment_points
    )

    X_train = np.array(signals)
    indices = np.random.choice(len(X_train), size=min(n_samples, len(X_train)), replace=False)
    print(f"Loaded {len(indices)} calibration samples from {len(X_train)} training segments.")
    return X_train[indices]


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def sigmoid(arr):
    """Numerically stable sigmoid applied element-wise."""
    return np.where(arr >= 0,
                    1 / (1 + np.exp(-arr)),
                    np.exp(arr) / (1 + np.exp(arr)))


def sliding_mean(arr, window_size=16):
    """Sliding mean over the first dimension of a 2D array with left zero-padding."""
    helper = np.zeros((arr.shape[0] + window_size, arr.shape[1]), dtype=arr.dtype)
    helper[window_size:, :] = arr
    cumsum = np.cumsum(helper, axis=0)
    sliding_sum = cumsum[window_size:, :] - cumsum[:-window_size, :]
    return sliding_sum / window_size


# ---------------------------------------------------------------------------
# Streaming inference
# ---------------------------------------------------------------------------

def run_streaming_inference(model, stream, timestep, sliding_window=16, apply_sigmoid=True):
    """Run streaming inference on a continuous vibration input (2D: time x channels).

    Steps through the stream in chunks of `timestep` samples, calls model.forward
    (Akida) or the model callable (TF), and optionally applies sliding mean and sigmoid.

    Args:
        model: Stateful Keras model or Akida model (already converted/mapped).
        stream (np.ndarray): Shape (n_time, n_channels). Raw vibration signal (int16).
        timestep (int): Samples per inference step (e.g. 256).
        sliding_window (int): Window size for temporal smoothing. 0 to disable.
        apply_sigmoid (bool): Whether to apply sigmoid to smoothed logits.

    Returns:
        preds_raw (np.ndarray): Shape (n_frames, n_faults) — raw logits per frame.
        preds_smooth (np.ndarray): Shape (n_frames, n_faults) — smoothed, optionally sigmoid.
    """
    import tensorflow as tf
    try:
        import akida
        in_akida = isinstance(model, akida.Model)
    except Exception:
        in_akida = False

    if not in_akida:
        model_func = tf.function(model)

    preds = []

    if stream.ndim == 1:
        stream = stream[:, np.newaxis]
    n_time = stream.shape[0]
    if n_time % timestep != 0:
        pad_length = timestep - (n_time % timestep)
        stream = np.pad(stream, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)

    for start in range(0, stream.shape[0], timestep):
        frame = stream[start:start + timestep]
        frame_in = np.expand_dims(frame, 0)  # (1, timestep, n_ch)
        if in_akida:
            out = model.forward(np.expand_dims(frame_in, axis=1).astype(np.int16))
        else:
            out = model_func(tf.convert_to_tensor(frame_in.astype(np.float32)))
        out = np.squeeze(np.asarray(out))
        if out.ndim == 1:
            out = out[np.newaxis, :]
        preds.append(out)

    preds_raw = np.concatenate(preds, axis=0)
    if sliding_window > 0:
        preds_smooth = sliding_mean(preds_raw, window_size=sliding_window)
    else:
        preds_smooth = preds_raw.copy()
    if apply_sigmoid:
        preds_smooth = sigmoid(preds_smooth)
    return preds_raw, preds_smooth


def segment_predictions(preds_raw, n_segments, segments_per_sample, threshold=0.5):
    """Aggregate per-frame logits into per-segment multi-label predictions.

    Args:
        preds_raw (np.ndarray): Shape (n_frames_total, n_faults) raw logits.
        n_segments (int): Number of complete segments.
        segments_per_sample (int): Frames per segment (length // timestep).
        threshold (float): Sigmoid threshold for binary decision. Default 0.5.

    Returns:
        seg_probs (np.ndarray): Shape (n_segments, n_faults) — per-segment sigmoid probs.
        seg_preds (np.ndarray): Shape (n_segments, n_faults) — binary predictions.
    """
    seg_probs = []
    for i in range(n_segments):
        start = i * segments_per_sample
        end = (i + 1) * segments_per_sample
        mean_logits = preds_raw[start:end].mean(axis=0)
        seg_probs.append(sigmoid(mean_logits))
    seg_probs = np.array(seg_probs)
    seg_preds = (seg_probs >= threshold).astype(int)
    return seg_probs, seg_preds


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_stateful_multilabel(model, X_test, y_test, length=4096, timestep=256,
                                 in_akida=None, threshold=0.5, verbose=0):
    """Evaluate a stateful model frame-by-frame on the full test set.

    Each segment is fed one frame at a time; the model's internal state carries
    temporal context. State is reset between segments. Per-segment prediction is
    the mean of the raw logits across all frames, then sigmoid + threshold.

    Args:
        model: Stateful Keras model or Akida model.
        X_test (np.ndarray): Shape (n_samples, length, n_channels) int16.
        y_test (np.ndarray): Shape (n_samples, 4) float32 multi-label ground truth.
        length (int): Segment length in samples.
        timestep (int): Frame size in samples.
        in_akida (bool, optional): Override Akida detection. None = auto-detect.
        threshold (float): Decision threshold for sigmoid outputs.

    Returns:
        dict: AUROC macro/micro, AUPRC macro, exact match accuracy, Hamming score,
              F1 macro/micro/weighted, per-fault AUROC.
    """
    import tensorflow as tf
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

    try:
        import akida
        _in_akida = isinstance(model, akida.Model)
    except ImportError:
        _in_akida = False
    if in_akida is not None:
        _in_akida = in_akida

    if not _in_akida:
        model_func = tf.function(model)

    segments_per_sample = length // timestep
    N = X_test.shape[0]
    if X_test.ndim == 2:
        X_test = X_test[..., np.newaxis]

    all_probs = []
    all_labels = []

    try:
        from tqdm.auto import tqdm as _tqdm
        _iter = _tqdm(range(N), desc="Evaluating", unit="seg") if verbose else range(N)
    except ImportError:
        _iter = range(N)

    for i in _iter:
        seg = X_test[i]  # (length, n_ch)
        logit_accum = np.zeros(y_test.shape[1], dtype=np.float64)

        for t in range(segments_per_sample):
            frame = seg[t * timestep:(t + 1) * timestep]
            frame_in = np.expand_dims(frame, 0).astype(np.float32)  # (1, timestep, n_ch)
            if _in_akida:
                frame_akida = np.expand_dims(frame_in, axis=1).astype(np.int16)
                out = model.forward(frame_akida)
            else:
                out = model_func(tf.convert_to_tensor(frame_in))
            out = np.squeeze(np.asarray(out))
            logit_accum += out

        mean_logits = logit_accum / segments_per_sample
        all_probs.append(sigmoid(mean_logits))
        all_labels.append(y_test[i])

        # Reset state for next segment
        if _in_akida:
            try:
                import akida as _akida
                model = _akida.Model(model.layers)
            except Exception:
                pass
        else:
            model.reset_states()

    probs = np.array(all_probs)
    labels = np.array(all_labels)
    preds_binary = (probs >= threshold).astype(int)

    metrics = {}

    # AUROC
    try:
        metrics['auroc_macro'] = roc_auc_score(labels, probs, average='macro')
        metrics['auroc_micro'] = roc_auc_score(labels, probs, average='micro')
    except Exception:
        metrics['auroc_macro'] = float('nan')
        metrics['auroc_micro'] = float('nan')

    # AUPRC
    try:
        metrics['auprc_macro'] = average_precision_score(labels, probs, average='macro')
    except Exception:
        metrics['auprc_macro'] = float('nan')

    # Exact match accuracy
    metrics['exact_match'] = accuracy_score(labels, preds_binary)

    # Hamming score
    metrics['hamming_score'] = float(np.mean(labels == preds_binary))

    # F1
    metrics['f1_macro'] = f1_score(labels, preds_binary, average='macro', zero_division=0)
    metrics['f1_micro'] = f1_score(labels, preds_binary, average='micro', zero_division=0)
    metrics['f1_weighted'] = f1_score(labels, preds_binary, average='weighted', zero_division=0)

    # Per-fault AUROC
    from sklearn.metrics import roc_auc_score as _auroc
    for i, name in enumerate(FAULT_NAMES):
        try:
            metrics[f'auroc_{name}'] = _auroc(labels[:, i], probs[:, i])
        except Exception:
            metrics[f'auroc_{name}'] = float('nan')

    return metrics


def print_metrics_report(metrics, title="Evaluation Results"):
    """Print a compact evaluation report (AUROC + per-label accuracy) for multi-label metrics."""
    w = 60
    print("=" * w)
    print(title)
    print("=" * w)
    print("\nPrimary Metrics:")
    print(f"  Macro AUROC:        {metrics.get('auroc_macro', float('nan')):.4f}")
    print(f"  Micro AUROC:        {metrics.get('auroc_micro', float('nan')):.4f}")
    print(f"  Hamming Score:      {metrics.get('hamming_score', float('nan')):.4f}   (per-label accuracy)")
    print("\nPer-Fault AUROC:")
    for name in FAULT_NAMES:
        print(f"  {name:12s}:    {metrics.get(f'auroc_{name}', float('nan')):.4f}")
    print("=" * w)


def print_param_summary(model, model_stateful, model_quantized):
    """Print a parameter-count and memory summary across the pipeline stages.

    Reports parameters and rough memory footprint for the kernelized (training),
    stateful (streaming), and quantized INT8 (deployment) forms of the model,
    showing how the model shrinks on the path to hardware.

    Args:
        model: Kernelized (full-sequence) Keras model.
        model_stateful: Stateful streaming Keras model.
        model_quantized: Quantized stateful Keras model.
    """
    k_params = model.count_params()
    s_params = model_stateful.count_params()
    q_params = model_quantized.count_params()

    print(f"{'Stage':<30} {'Parameters':>12} {'Float (KB)':>12} {'INT8 (KB)':>12}")
    print("─" * 68)
    print(f"{'Kernelized (training)':<30} {k_params:>12,} {k_params * 4 / 1024:>12.1f} {'—':>12}")
    print(f"{'Stateful (streaming)':<30} {s_params:>12,} {s_params * 4 / 1024:>12.1f} {'—':>12}")
    print(f"{'Quantized INT8 (stateful)':<30} {q_params:>12,} {q_params * 4 / 1024:>12.1f} {q_params / 1024:>12.1f}")


def plot_healthy_vs_faulty(X_test, y_test, length, sample_rate=SAMPLE_RATE,
                           fault_names=FAULT_NAMES):
    """Plot one healthy segment above one faulty segment for a visual sanity check.

    Picks the first healthy (all-zero label) segment and the first segment with any
    fault active, and draws their raw int16 waveforms on a shared time axis so the
    difference in vibration signature is visible at a glance.

    Args:
        X_test (np.ndarray): Test segments, shape (n, length, 1) int16.
        y_test (np.ndarray): Multi-label targets, shape (n, n_faults).
        length (int): Samples per segment (x-axis length).
        sample_rate (int): Sampling rate in Hz, used for the time axis.
        fault_names (list): Fault names indexed by column of y_test.
    """
    import matplotlib.pyplot as plt

    idx_healthy = int(np.where(y_test.sum(axis=1) == 0)[0][0])
    idx_faulty = int(np.where(y_test.sum(axis=1) > 0)[0][0])
    t_ms = np.arange(length) / sample_rate * 1000
    active = [fault_names[i] for i, v in enumerate(y_test[idx_faulty]) if v > 0.5]

    fig, axs = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    axs[0].plot(t_ms, X_test[idx_healthy, :, 0], color='#2d5a27', linewidth=0.6)
    axs[0].set_title('Healthy bearing')
    axs[0].set_ylabel('Amplitude (int16)')
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(t_ms, X_test[idx_faulty, :, 0], color='#e74c3c', linewidth=0.6)
    axs[1].set_title(f"Faulty bearing — faults: {', '.join(active)}")
    axs[1].set_ylabel('Amplitude (int16)')
    axs[1].set_xlabel('Time (ms)')
    axs[1].grid(True, alpha=0.3)

    duration_s = length / sample_rate
    fig.suptitle(f'Vibration signal: healthy vs faulty '
                 f'({sample_rate / 1000:.0f} kHz, {length} samples ~{duration_s:.0f} s)', fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_label_distribution(y_test, title="Label Distribution"):
    """Bar chart showing how many samples have each fault type active,
    plus how many samples are healthy (all-zero label).

    Args:
        y_test (np.ndarray): Shape (n_samples, 4) multi-label array.
        title (str): Plot title.
    """
    import matplotlib.pyplot as plt

    labels = np.asarray(y_test)
    counts = labels.sum(axis=0)
    n_healthy = np.sum(labels.sum(axis=1) == 0)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[0].bar(FAULT_NAMES, counts, color=FAULT_COLORS, edgecolor='white')
    axs[0].set_xlabel('Fault Type')
    axs[0].set_ylabel('Number of samples with fault active')
    axs[0].set_title('Active fault counts')
    axs[0].grid(True, alpha=0.3, axis='y')
    for i, c in enumerate(counts):
        axs[0].text(i, c + 0.5, str(int(c)), ha='center', va='bottom', fontsize=9)

    health_labels = ['Healthy', 'Any Fault']
    health_counts = [int(n_healthy), int(len(labels) - n_healthy)]
    axs[1].bar(health_labels, health_counts, color=['#2ecc71', '#e74c3c'], edgecolor='white')
    axs[1].set_ylabel('Number of samples')
    axs[1].set_title('Healthy vs faulty samples')
    axs[1].grid(True, alpha=0.3, axis='y')
    for i, c in enumerate(health_counts):
        axs[1].text(i, c + 0.5, str(c), ha='center', va='bottom', fontsize=9)

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Plotting — streaming demo
# ---------------------------------------------------------------------------

def plot_streaming_demo(signal, preds_raw, seg_probs, seg_preds, true_labels,
                        length, timestep, n_segments, sample_rate=SAMPLE_RATE,
                        fault_names=FAULT_NAMES):
    """3-panel streaming demo plot:
      1. Raw vibration waveform with segment boundaries
      2. Per-fault probability heatmap over time
      3. Predicted vs true labels per segment (dot grid)

    Args:
        signal (np.ndarray): 1D or (n,1) vibration stream.
        preds_raw (np.ndarray): (n_frames, n_faults) raw logits.
        seg_probs (np.ndarray): (n_segments, n_faults) per-segment sigmoid probs.
        seg_preds (np.ndarray): (n_segments, n_faults) binary predictions.
        true_labels (np.ndarray): (n_segments, 4) ground truth multi-label.
        length (int): Segment length in samples.
        timestep (int): Frame size in samples.
        n_segments (int): Number of segments.
        sample_rate (int): Sampling rate in Hz.
        fault_names (list): Fault type names.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    segments_per_sample = length // timestep
    sig_1d = np.squeeze(signal)
    total_duration_ms = n_segments * length / sample_rate * 1000
    frame_duration_ms = timestep / sample_rate * 1000
    n_frames = preds_raw.shape[0]

    fig, axs = plt.subplots(3, 1, figsize=(14, 9), constrained_layout=True)

    # Panel 1: waveform
    t_sig_ms = np.arange(len(sig_1d)) / sample_rate * 1000
    axs[0].plot(t_sig_ms, sig_1d, color='#555555', linewidth=0.5, alpha=0.85)
    for i in range(1, n_segments):
        axs[0].axvline(i * length / sample_rate * 1000, color='#3498db', linestyle='--', alpha=0.6)
    axs[0].set_ylabel('Amplitude')
    axs[0].set_title('Vibration stream (raw sensor input)')
    axs[0].set_xlim(0, total_duration_ms)
    axs[0].grid(True, alpha=0.2)

    # Panel 2: probability heatmap (n_faults x n_frames)
    probs_smooth = sigmoid(preds_raw)  # (n_frames, n_faults)
    im = axs[1].imshow(
        probs_smooth.T,
        aspect='auto',
        interpolation='nearest',
        extent=[0, total_duration_ms, -0.5, len(fault_names) - 0.5],
        vmin=0, vmax=1,
        cmap='Blues',
        origin='lower',
    )
    for i in range(1, n_segments):
        axs[1].axvline(i * length / sample_rate * 1000, color='white', linestyle='--', alpha=0.6)
    axs[1].set_yticks(range(len(fault_names)))
    axs[1].set_yticklabels(fault_names)
    axs[1].set_ylabel('Fault type')
    axs[1].set_title('Per-fault probability over time (sigmoid of logits)')
    axs[1].set_xlim(0, total_duration_ms)
    plt.colorbar(im, ax=axs[1], fraction=0.015, pad=0.01, label='Probability')

    # Panel 3: predicted vs true per segment
    seg_centers_ms = [(i + 0.5) * length / sample_rate * 1000 for i in range(n_segments)]
    for fi, (fname, fcolor) in enumerate(zip(fault_names, FAULT_COLORS)):
        for si in range(n_segments):
            pred = seg_preds[si, fi]
            true = int(true_labels[si, fi])
            marker = 'o' if pred == 1 else 'x'
            match = (pred == true)
            edge_color = '#2ecc71' if match else '#e74c3c'
            axs[2].scatter(
                seg_centers_ms[si], fi,
                marker=marker, s=80,
                color=fcolor, edgecolors=edge_color, linewidths=1.5, alpha=0.85,
                zorder=3,
            )
            if true == 1:
                axs[2].scatter(
                    seg_centers_ms[si], fi,
                    marker='s', s=130,
                    color='none', edgecolors='black', linewidths=1.0, alpha=0.4,
                    zorder=2,
                )

    for i in range(1, n_segments):
        axs[2].axvline(i * length / sample_rate * 1000, color='#bdc3c7', linestyle='--', alpha=0.6)

    axs[2].set_yticks(range(len(fault_names)))
    axs[2].set_yticklabels(fault_names)
    axs[2].set_ylabel('Fault type')
    axs[2].set_xlabel('Time (ms)')
    axs[2].set_title('Predicted (circles=fault, x=healthy) vs true (square outline) per segment')
    axs[2].set_xlim(0, total_duration_ms)
    axs[2].set_ylim(-0.8, len(fault_names) - 0.2)
    axs[2].grid(True, alpha=0.2, axis='x')

    correct_patch = mpatches.Patch(edgecolor='#2ecc71', facecolor='none', label='Correct prediction')
    wrong_patch = mpatches.Patch(edgecolor='#e74c3c', facecolor='none', label='Wrong prediction')
    true_patch = mpatches.Patch(edgecolor='black', facecolor='none', label='True fault present (square)')
    axs[2].legend(handles=[correct_patch, wrong_patch, true_patch],
                  loc='upper right', fontsize=8, framealpha=0.7)

    plt.show()


def plot_streaming_debug(signal, preds_raw, preds_smooth, sliding_window=16):
    """3-panel debug view: raw vibration waveform, raw logit heatmap, smoothed probability heatmap.

    Args:
        signal (np.ndarray): 1D vibration stream.
        preds_raw (np.ndarray): (n_frames, n_faults) raw logits.
        preds_smooth (np.ndarray): (n_frames, n_faults) smoothed probabilities.
        sliding_window (int): Window size used for smoothing.
    """
    import matplotlib.pyplot as plt

    sig_1d = np.squeeze(signal)
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    axs[0].plot(sig_1d, color='#555555', linewidth=0.5)
    axs[0].set_title('Vibration stream input')
    axs[0].set_xlabel('Input samples')
    axs[0].set_ylabel('Amplitude')

    axs[1].imshow(preds_raw.T, aspect='auto', interpolation='nearest', cmap='RdBu_r')
    axs[1].set_title('Raw logits per frame')
    axs[1].set_yticks(range(len(FAULT_NAMES)))
    axs[1].set_yticklabels(FAULT_NAMES)
    axs[1].set_xlabel('Frame index')

    axs[2].imshow(preds_smooth.T, aspect='auto', interpolation='nearest', cmap='Blues',
                  vmin=0, vmax=1)
    axs[2].set_title(f'Smoothed probabilities (sliding mean window={sliding_window} + sigmoid)')
    axs[2].set_yticks(range(len(FAULT_NAMES)))
    axs[2].set_yticklabels(FAULT_NAMES)
    axs[2].set_xlabel('Frame index')

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Plotting — evaluation comparison
# ---------------------------------------------------------------------------

def plot_evaluation_comparison(metrics_dict, primary_metrics=None):
    """Side-by-side grouped bar chart comparing metrics across float/quantized/Akida variants.

    Args:
        metrics_dict (dict): Keys are model variant names (e.g. 'Float', 'INT8', 'Akida'),
                             values are metric dicts from evaluate_stateful_multilabel.
        primary_metrics (list, optional): Metric keys to plot. Defaults to main AUROC/F1 metrics.
    """
    import matplotlib.pyplot as plt

    if primary_metrics is None:
        primary_metrics = ['auroc_macro', 'auprc_macro', 'f1_macro', 'exact_match', 'hamming_score']

    variants = list(metrics_dict.keys())
    n_metrics = len(primary_metrics)
    n_variants = len(variants)
    x = np.arange(n_metrics)
    width = 0.8 / n_variants
    bar_colors = ['#2d5a27', '#3498db', '#e74c3c', '#f39c12']

    fig, ax = plt.subplots(figsize=(10, 5))
    for vi, (variant, color) in enumerate(zip(variants, bar_colors)):
        vals = [metrics_dict[variant].get(m, 0) for m in primary_metrics]
        offset = (vi - n_variants / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=variant, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '\n') for m in primary_metrics], fontsize=9)
    ax.set_ylabel('Score')
    ax.set_title('Evaluation metrics: Float vs Akida (Pico)')
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def plot_per_fault_auroc(metrics_dict):
    """Per-fault AUROC comparison across model variants.

    Args:
        metrics_dict (dict): Keys are model variant names, values are metric dicts.
    """
    import matplotlib.pyplot as plt

    variants = list(metrics_dict.keys())
    n_faults = len(FAULT_NAMES)
    x = np.arange(n_faults)
    n_variants = len(variants)
    width = 0.8 / n_variants
    bar_colors = ['#2d5a27', '#3498db', '#e74c3c', '#f39c12']

    fig, ax = plt.subplots(figsize=(9, 4))
    for vi, (variant, color) in enumerate(zip(variants, bar_colors)):
        vals = [metrics_dict[variant].get(f'auroc_{fn}', 0) for fn in FAULT_NAMES]
        offset = (vi - n_variants / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=variant, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(FAULT_NAMES)
    ax.set_ylabel('AUROC')
    ax.set_title('Per-fault AUROC: Float vs Akida (Pico)')
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Plotting — hardware performance
# ---------------------------------------------------------------------------

def plot_power_estimates(clocks_mhz, duty_power_uw):
    """Plot duty-cycle average power vs clock speed."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(clocks_mhz, duty_power_uw, 's-', color='#3498db')
    ax.set_xlabel('Clock (MHz)')
    ax.set_ylabel('Duty-cycle power (uW)')
    ax.set_title('Average power - always-on condition monitoring')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_duty_cycle_explanation(inference_clk, buffer_ms=6.1, clocks_mhz=(25, 100),
                                dynamic_50mhz_uw=550, leakage_uw=14):
    """Power vs time plot for two clock speeds, illustrating the duty-cycle tradeoff.

    Args:
        inference_clk: Hardware cycles per inference (from model_akida).
        buffer_ms (float): Frame period in ms (256 / 42000 * 1000 ~ 6.1 ms).
        clocks_mhz (tuple): Two clock speeds to compare.
        dynamic_50mhz_uw (float): Dynamic power at 50 MHz (uW).
        leakage_uw (float): Leakage power (uW).
    """
    import matplotlib.pyplot as plt

    ref_mhz = 50
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    for ax, f_mhz in zip(axs, clocks_mhz):
        f_hz = f_mhz * 1e6
        active_ms = (inference_clk / f_hz) * 1000
        idle_ms = max(0, buffer_ms - active_ms)
        scale = f_mhz / ref_mhz
        dynamic_uw = dynamic_50mhz_uw * scale
        total_uw = dynamic_uw + leakage_uw
        duty = min(active_ms / buffer_ms, 1.0)
        avg_power = total_uw * duty + leakage_uw * (1 - duty)

        t = [0, active_ms, active_ms, buffer_ms]
        p = [total_uw, total_uw, leakage_uw, leakage_uw]
        ax.step(t, p, where='post', color='#3498db', linewidth=2, label='Power')
        ax.fill_between(t, p, alpha=0.3, step='post', color='#3498db')
        ax.set_xlim(0, buffer_ms + 0.5)
        ax.set_ylim(0, total_uw * 1.15)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Power (uW)')
        ax.set_title(f'{f_mhz} MHz\n~{avg_power:.1f} uW avg')
        ax.axvline(buffer_ms, color='#7f8c8d', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Power vs time: duty cycle - same energy per inference, different clocks', fontsize=11)
    f0_hz = clocks_mhz[0] * 1e6
    act0_ms = (inference_clk / f0_hz) * 1000
    total0_uw = dynamic_50mhz_uw * (clocks_mhz[0] / ref_mhz) + leakage_uw
    duty0 = min(act0_ms / buffer_ms, 1.0)
    avg0_uw = total0_uw * duty0 + leakage_uw * (1 - duty0)
    e_per_frame = avg0_uw * (buffer_ms / 1000)
    takeaway = (
        f"Takeaway: ~{avg0_uw:.0f} uW average power for always-on real-time fault monitoring. "
        f"~{e_per_frame:.3f} uJ per frame (constant with clock). "
        f"Frame period = {buffer_ms:.1f} ms (256 samples @ {SAMPLE_RATE/1000:.0f} kHz)."
    )
    fig.text(0.5, 0.02, textwrap.fill(takeaway, width=90), ha='center', va='bottom', fontsize=9,
             transform=fig.transFigure, color='#555555')
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show()


def plot_throughput_vs_clock(clocks_mhz, inference_clk, buffer_ms=6.1):
    """Max FPS vs clock with required FPS line for real-time streaming.

    Args:
        clocks_mhz (np.ndarray): Clock speeds in MHz.
        inference_clk (int): Hardware cycles per inference.
        buffer_ms (float): Frame period in ms (6.1 ms for 256 samples @ 42 kHz).
    """
    import matplotlib.pyplot as plt

    max_fps = [f_mhz * 1e6 / inference_clk for f_mhz in clocks_mhz]
    required_fps = 1000 / buffer_ms

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(clocks_mhz, max_fps, 'o-', color='#2d5a27', label='Max FPS')
    ax.axhline(required_fps, color='#e74c3c', linestyle='--', alpha=0.8,
               label=f'Required ({required_fps:.0f} fps for {buffer_ms:.1f} ms frame)')
    ax.set_xlabel('Clock (MHz)')
    ax.set_ylabel('Throughput (FPS)')
    ax.set_title('Throughput vs Clock - can the model keep up with the sensor?')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(np.asarray(clocks_mhz).min() - 2, np.asarray(clocks_mhz).max() + 2)
    takeaway = (
        f"Takeaway: Stay above the required FPS line ({required_fps:.0f} fps for {buffer_ms:.1f} ms "
        f"frame at {SAMPLE_RATE/1000:.0f} kHz) for real-time streaming. "
        "Higher clock speeds increase throughput headroom."
    )
    fig.text(0.5, 0.02, textwrap.fill(takeaway, width=90), ha='center', va='bottom', fontsize=9,
             transform=fig.transFigure, color='#555555')
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.show()


def plot_latency_vs_clock(clocks_mhz, latency_ms, buffer_ms=6.1):
    """Inference latency vs clock with real-time deadline line.

    Args:
        clocks_mhz (np.ndarray): Clock speeds in MHz.
        latency_ms (np.ndarray): Latency in ms per clock speed.
        buffer_ms (float): Frame period in ms.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(clocks_mhz, latency_ms, 'o-', color='#2d5a27', label='Inference latency')
    ax.axhline(buffer_ms, color='#e74c3c', linestyle='--', alpha=0.8,
               label=f'Frame deadline ({buffer_ms:.1f} ms)')
    ax.set_xlabel('Clock (MHz)')
    ax.set_ylabel('Inference latency (ms)')
    ax.set_title('Latency vs Clock')
    ax.legend()
    ax.grid(True, alpha=0.3)
    takeaway = (
        f"Takeaway: Keep latency below the frame deadline ({buffer_ms:.1f} ms) to process "
        "every 256-sample vibration frame in real time."
    )
    fig.text(0.5, 0.02, textwrap.fill(takeaway, width=90), ha='center', va='bottom', fontsize=9,
             transform=fig.transFigure, color='#555555')
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.show()


# ---------------------------------------------------------------------------
# Decision latency — accuracy vs. number of accumulated chunks
# ---------------------------------------------------------------------------

def sweep_decision_latency(model, X_test, y_test, timestep=256, cutoffs=None,
                           in_akida=None, sample_rate=SAMPLE_RATE):
    """Measure accuracy as a function of how many streaming chunks are accumulated
    before a prediction is read out.

    The stateful model emits one output per ``timestep``-sample chunk. A prediction is
    formed by averaging the per-chunk logits accumulated so far, then applying a sigmoid
    and a 0.5 threshold, so a prediction can be produced after any number of chunks.
    Accumulating more chunks uses more temporal evidence but increases the time-to-decision.
    State is reset between segments.

    Args:
        model: stateful Keras model or Akida model.
        X_test (np.ndarray): shape (n_samples, length, 1) int16 segments.
        y_test (np.ndarray): shape (n_samples, n_classes) float32 multi-label ground truth.
        timestep (int): chunk size in samples.
        cutoffs (list of int, optional): chunk counts at which to evaluate. Defaults to a
            spread of values up to the full segment length.
        in_akida (bool, optional): override Akida detection. None auto-detects from model type.
        sample_rate (int): samples per second, used to convert chunk counts to milliseconds.

    Returns:
        dict: 'chunks', 'latency_ms', 'auroc', 'hamming', 'f1' -- lists aligned by cutoff.
    """
    from sklearn.metrics import roc_auc_score, f1_score

    try:
        import akida
        _in_akida = isinstance(model, akida.Model)
    except ImportError:
        _in_akida = False
    if in_akida is not None:
        _in_akida = in_akida
    if not _in_akida:
        import tensorflow as tf
        model_func = tf.function(model)

    if X_test.ndim == 2:
        X_test = X_test[..., np.newaxis]
    n_samples, seg_len = X_test.shape[0], X_test.shape[1]
    n_chunks = seg_len // timestep
    n_classes = y_test.shape[1]

    if cutoffs is None:
        cutoffs = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, n_chunks]
    cutoffs = sorted({c for c in cutoffs if 1 <= c <= n_chunks})
    cutset = set(cutoffs)

    probs = {c: np.zeros((n_samples, n_classes)) for c in cutoffs}
    for i in range(n_samples):
        seg = X_test[i]
        running = np.zeros(n_classes, dtype=np.float64)
        for t in range(n_chunks):
            frame = np.expand_dims(seg[t * timestep:(t + 1) * timestep], 0).astype(np.float32)
            if _in_akida:
                out = model.forward(np.expand_dims(frame, axis=1).astype(np.int16))
            else:
                import tensorflow as tf
                out = model_func(tf.convert_to_tensor(frame))
            running += np.squeeze(np.asarray(out))
            m = t + 1
            if m in cutset:
                probs[m][i] = sigmoid(running / m)
        # Reset state between segments.
        if _in_akida:
            try:
                import akida as _akida
                model = _akida.Model(model.layers)
            except Exception:
                pass
        else:
            model.reset_states()

    result = {'chunks': [], 'latency_ms': [], 'auroc': [], 'hamming': [], 'f1': []}
    for c in cutoffs:
        p = probs[c]
        pred = (p >= 0.5).astype(int)
        result['chunks'].append(c)
        result['latency_ms'].append(c * timestep / sample_rate * 1000.0)
        try:
            result['auroc'].append(roc_auc_score(y_test, p, average='macro'))
        except ValueError:
            result['auroc'].append(float('nan'))
        result['hamming'].append(float(np.mean(pred == y_test.astype(int))))
        result['f1'].append(f1_score(y_test, pred, average='macro', zero_division=0))
    return result


def plot_accuracy_vs_latency(result, train_window_ms=None):
    """Plot macro AUROC and Hamming score versus decision latency.

    Args:
        result (dict): output of :func:`sweep_decision_latency`.
        train_window_ms (float, optional): if given, marks the model's training window
            on the latency axis.
    """
    import matplotlib.pyplot as plt

    lat = result['latency_ms']
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(lat, result['auroc'], 'o-', color='#2d5a27', label='Macro AUROC')
    ax.plot(lat, result['hamming'], 's--', color='#3498db', label='Hamming score')
    if train_window_ms is not None:
        ax.axvline(train_window_ms, color='#888888', ls=':', lw=1)
        ymin = min(min(result['auroc']), min(result['hamming']))
        ax.text(train_window_ms, ymin, f' training window ≈ {train_window_ms:.0f} ms',
                rotation=90, va='bottom', ha='left', fontsize=8, color='#666666')
    ax.set_xlabel('Decision latency — vibration data accumulated (ms)')
    ax.set_ylabel('Score')
    ax.set_title('Accuracy vs. decision latency')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
