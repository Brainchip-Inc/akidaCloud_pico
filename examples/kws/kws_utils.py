"""
Utilities for the KWS demo notebook: data loading, streaming inference helpers,
sliding mean, and plotting. Keeps the notebook focused on the main pipeline.
"""

import os
import textwrap

import numpy as np

# SC10 (Speech Commands 10-class) label order
LABEL_NAMES = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
SAMPLE_RATE = 16000  # Speech Commands is 16 kHz


def load_test_data(x_path, y_path, length=16384):
    """
    Load test waveforms and labels from pre-saved .npy files.
    Returns X_test shape (n_samples, length, 1), y_test shape (n_samples,).
    """
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Test data not found: {x_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Labels not found: {y_path}")
    X_test = np.load(x_path)
    y_test = np.load(y_path)
    if len(X_test) != len(y_test):
        raise ValueError(
            f"X and y length mismatch: X_test has {len(X_test)} samples, "
            f"y_test has {len(y_test)}. Ensure both files correspond to the same dataset."
        )
    if X_test.ndim == 2:
        X_test = X_test[..., np.newaxis]
    return X_test, y_test


def _compute_trim_bounds(
    wav,
    n,
    sample_rate=16000,
    energy_win_ms=20,
    threshold_frac=0.05,
    margin_ms=30,
    min_speech_ms=200,
):
    """Return (start_s, end_s) sample indices for trimming one waveform. Used by trim_silence_and_pad and get_trim_bounds."""
    win = max(1, int(sample_rate * energy_win_ms / 1000))
    hop = max(1, win // 2)
    n_frames = (n - win) // hop + 1
    if n_frames <= 0:
        return 0, n

    energies = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        chunk = wav[start : start + win]
        energies[i] = np.sqrt(np.mean(chunk * chunk))

    max_energy = np.max(energies)
    if max_energy <= 0:
        return 0, n

    threshold = float(threshold_frac) * max_energy
    above = energies > threshold
    if not np.any(above):
        return 0, n

    first_above = np.flatnonzero(above)[0]
    last_above = np.flatnonzero(above)[-1]
    start_s = first_above * hop
    end_s = min(n, last_above * hop + win)
    margin_samp = int(sample_rate * margin_ms / 1000)
    min_samp = int(sample_rate * min_speech_ms / 1000)
    start_s = max(0, start_s - margin_samp)
    end_s = min(n, end_s + margin_samp)
    if end_s - start_s < min_samp:
        start_s = 0
        end_s = n
    return start_s, end_s


def get_trim_bounds(
    waveform,
    sample_rate=16000,
    energy_win_ms=20,
    threshold_frac=0.05,
    margin_ms=30,
    min_speech_ms=200,
):
    """
    Get the (start, end) sample indices that trim_silence_and_pad would use for this clip.
    Useful to check trimming without modifying data. Returns (start_s, end_s).
    """
    wav = np.squeeze(waveform).astype(np.float64)
    n = len(wav)
    if n == 0:
        return 0, 0
    return _compute_trim_bounds(
        wav, n,
        sample_rate=sample_rate,
        energy_win_ms=energy_win_ms,
        threshold_frac=threshold_frac,
        margin_ms=margin_ms,
        min_speech_ms=min_speech_ms,
    )


def trim_silence_and_pad(
    waveform,
    length=16384,
    sample_rate=16000,
    energy_win_ms=20,
    threshold_frac=0.05,
    margin_ms=30,
    min_speech_ms=200,
):
    """
    Trim leading and trailing silence so the word starts at t=0, then pad to fixed length.
    Uses per-clip energy threshold so quiet and loud clips both trim sensibly.
    Use the returned waveform for eval so "first N frames" = first N frames after word start.

    Args:
        waveform: (samples,) or (samples, 1). One clip, typically 16384 samples.
        length: Output length (samples). Padded with zeros at the end.
        sample_rate: Used to convert margin_ms and min_speech_ms to samples.
        energy_win_ms: Window length (ms) for computing frame energy (RMS).
        threshold_frac: Speech start/end when frame energy > this fraction of max energy in clip.
        margin_ms: Safety margin (ms) before detected start and after detected end (don't cut into word).
        min_speech_ms: If detected speech is shorter than this (ms), skip trimming and return original.

    Returns:
        Padded waveform shape (length,) or (length, 1) to match input ndim. Word is at the start.
    """
    wav = np.squeeze(waveform).astype(np.float64)
    n = len(wav)
    if n == 0:
        out = np.zeros(length, dtype=np.float32)
        return out[..., np.newaxis] if np.asarray(waveform).ndim > 1 else out

    start_s, end_s = _compute_trim_bounds(
        wav, n,
        sample_rate=sample_rate,
        energy_win_ms=energy_win_ms,
        threshold_frac=threshold_frac,
        margin_ms=margin_ms,
        min_speech_ms=min_speech_ms,
    )

    trimmed = wav[start_s:end_s].astype(np.float32)
    if len(trimmed) >= length:
        out = trimmed[:length].copy()
    else:
        out = np.zeros(length, dtype=np.float32)
        out[: len(trimmed)] = trimmed

    if np.asarray(waveform).ndim > 1:
        out = out[..., np.newaxis]
    return out


def trim_silence_dataset(
    X_test,
    length=16384,
    sample_rate=16000,
    energy_win_ms=20,
    threshold_frac=0.05,
    margin_ms=30,
    min_speech_ms=200,
):
    """
    Trim leading/trailing silence for each clip so word start is at t=0, then pad to fixed length.
    Returns X_trimmed with same shape as X_test; labels (y_test) are unchanged.
    Use for fast-response eval: pass X_trimmed to build_stateful_eval_dataset / evaluate_stateful_full.
    """
    X_test = np.asarray(X_test)
    if X_test.ndim == 2:
        X_test = X_test[..., np.newaxis]
    n = X_test.shape[0]
    out = np.zeros((n, length, X_test.shape[2]), dtype=np.float32)
    for i in range(n):
        out[i] = trim_silence_and_pad(
            X_test[i],
            length=length,
            sample_rate=sample_rate,
            energy_win_ms=energy_win_ms,
            threshold_frac=threshold_frac,
            margin_ms=margin_ms,
            min_speech_ms=min_speech_ms,
        )
    return out


def plot_trim_check(
    X_test,
    y_test=None,
    indices=None,
    n_show=5,
    sample_rate=16000,
    energy_win_ms=20,
    threshold_frac=0.05,
    margin_ms=30,
    min_speech_ms=200,
):
    """
    Plot original waveforms with trim boundaries to check for bad trimming.
    Use after tuning trim params: if the vertical lines cut into the word or leave
    a lot of silence, adjust threshold_frac, margin_ms, or min_speech_ms.

    Args:
        X_test: (n, samples, 1) or (n, samples). Clips to inspect.
        y_test: Optional (n,) labels; if provided and LABEL_NAMES exists, subplot title shows keyword.
        indices: Optional list of clip indices to plot. If None, n_show random clips are chosen.
        n_show: Number of clips to plot when indices is None.
        sample_rate, energy_win_ms, threshold_frac, margin_ms, min_speech_ms: Same as trim_silence_and_pad.
    """
    import matplotlib.pyplot as plt

    X_test = np.asarray(X_test)
    if X_test.ndim == 2:
        X_test = X_test[..., np.newaxis]
    n_clips = X_test.shape[0]
    if indices is None:
        rng = np.random.default_rng()
        indices = rng.choice(n_clips, size=min(n_show, n_clips), replace=False)
    else:
        indices = np.atleast_1d(indices)

    n_plot = len(indices)
    fig, axs = plt.subplots(n_plot, 2, figsize=(12, 2 * n_plot), squeeze=False)
    time_ms = np.arange(X_test.shape[1]) * 1000.0 / sample_rate

    for row, idx in enumerate(indices):
        wav = np.squeeze(X_test[idx]).astype(np.float64)
        start_s, end_s = get_trim_bounds(
            X_test[idx],
            sample_rate=sample_rate,
            energy_win_ms=energy_win_ms,
            threshold_frac=threshold_frac,
            margin_ms=margin_ms,
            min_speech_ms=min_speech_ms,
        )
        start_ms = start_s * 1000.0 / sample_rate
        end_ms = end_s * 1000.0 / sample_rate

        axs[row, 0].plot(time_ms, wav, color="gray", alpha=0.8)
        axs[row, 0].axvline(start_ms, color="green", linestyle="--", linewidth=1.5, label="trim start")
        axs[row, 0].axvline(end_ms, color="red", linestyle="--", linewidth=1.5, label="trim end")
        axs[row, 0].set_xlim(0, time_ms[-1])
        axs[row, 0].set_xlabel("Time (ms)")
        axs[row, 0].set_ylabel("Amplitude")
        title = f"Clip {idx}"
        if y_test is not None and idx < len(y_test):
            try:
                title += f" — {LABEL_NAMES[int(y_test[idx])]}"
            except (IndexError, ValueError):
                pass
        axs[row, 0].set_title(title)
        axs[row, 0].legend(loc="upper right", fontsize=8)
        axs[row, 0].grid(True, alpha=0.3)

        trimmed = trim_silence_and_pad(
            X_test[idx],
            length=X_test.shape[1],
            sample_rate=sample_rate,
            energy_win_ms=energy_win_ms,
            threshold_frac=threshold_frac,
            margin_ms=margin_ms,
            min_speech_ms=min_speech_ms,
        )
        t_trim = np.squeeze(trimmed)
        axs[row, 1].plot(time_ms, t_trim, color="steelblue", alpha=0.9)
        axs[row, 1].set_xlim(0, time_ms[-1])
        axs[row, 1].set_xlabel("Time (ms)")
        axs[row, 1].set_ylabel("Amplitude")
        axs[row, 1].set_title(f"Clip {idx} after trim + pad (word at start)")
        axs[row, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def sliding_mean(arr, window_size=64):
    """Sliding mean over the first dimension of a 2D array with left zero-padding."""
    helper = np.zeros((arr.shape[0] + window_size, arr.shape[1]), dtype=arr.dtype)
    helper[window_size:, :] = arr
    cumsum = np.cumsum(helper, axis=0)
    sliding_sum = cumsum[window_size:, :] - cumsum[:-window_size, :]
    return sliding_sum / window_size


def softmax(arr):
    """Numerically stable softmax over the last axis."""
    exp_arr = np.exp(arr - np.max(arr, axis=-1, keepdims=True))
    return exp_arr / np.sum(exp_arr, axis=-1, keepdims=True)


def run_streaming_inference(model, stream, timestep, sliding_window=64, apply_softmax=True):
    """
    Run streaming inference on a continuous input (2D: time, channels).
    Steps through the stream in chunks of size `timestep`, calls model.forward (Akida)
    or the model callable (TF), and optionally applies sliding mean and softmax.
    Returns (preds_raw, preds_smooth) both shape (n_frames, n_classes).
    """
    import tensorflow as tf
    try:
        import akida
        in_akida = isinstance(model, akida.Model)
    except Exception:
        in_akida = False

    if not in_akida:
        model_func = tf.function(model)

    n_classes = int(model.output_shape[-1])
    preds = []
    if stream.ndim == 1:
        stream = stream[:, np.newaxis]
    n_time = stream.shape[0]
    if n_time % timestep != 0:
        pad_length = timestep - (n_time % timestep)
        stream = np.pad(stream, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)

    for start in range(0, stream.shape[0], timestep):
        frame = np.expand_dims(stream[start : start + timestep], 0).astype(np.float32)
        if in_akida:
            out = model.forward(np.expand_dims(frame, axis=1).astype(np.int16))
        else:
            out = model_func(tf.convert_to_tensor(frame))
        out = np.squeeze(np.asarray(out))
        if out.ndim == 1:
            out = out[np.newaxis, :]
        preds.append(out)

    preds_raw = np.concatenate(preds, axis=0)
    preds_smooth = sliding_mean(preds_raw, window_size=sliding_window)
    if apply_softmax:
        preds_smooth = softmax(preds_smooth)
    return preds_raw, preds_smooth


def segment_accuracy(preds_raw, labels, length, timestep, n_segments):
    """
    Compute segment-wise accuracy: average logits per segment, then argmax vs labels.
    Returns (segment_preds, accuracy_percent).
    """
    segments_per_sample = length // timestep
    segment_preds = []
    for i in range(n_segments):
        start = i * segments_per_sample
        end = (i + 1) * segments_per_sample
        seg_mean_logits = preds_raw[start:end].mean(axis=0)
        segment_preds.append(np.argmax(seg_mean_logits))
    segment_preds = np.array(segment_preds)
    labels = np.asarray(labels)[:n_segments]
    accuracy = (segment_preds == labels).mean() * 100
    return segment_preds, accuracy


def plot_streaming_debug(
    input_stream, preds_raw, preds_smooth, sliding_window=64, apply_softmax=True
):
    """
    Plot 3-panel debug view: audio input, raw logits over time, sliding mean (+ softmax) over time.
    input_stream: (n_time,) or (n_time, n_ch); preds_raw/preds_smooth: (n_frames, n_classes).
    """
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 8))
    inp = np.squeeze(input_stream)
    if inp.ndim == 0:
        inp = inp[np.newaxis]
    axs[0].plot(inp)
    axs[0].set_xlim([0, len(inp)])
    axs[0].set_title('Audio Input (stream)')
    axs[0].set_xlabel('Input Timepoints')
    axs[1].imshow(preds_raw.T, aspect='auto', interpolation='none')
    axs[1].set_title('Raw outputs')
    axs[1].set_ylabel('Classes')
    axs[1].set_xlabel('Output Timepoints')
    axs[2].imshow(preds_smooth.T, aspect='auto', interpolation='none')
    axs[2].set_title(
        f'Sliding Mean: {sliding_window} points' + (' + Softmax' if apply_softmax else '')
    )
    axs[2].set_xlabel('Output Timepoints')
    plt.tight_layout()
    plt.show()


def plot_streaming_demo(
    audio_1d, preds_raw, segment_preds, labels_10,
    length, timestep, n_stream_samples, sample_rate=SAMPLE_RATE,
    label_names=LABEL_NAMES
):
    """
    Plot waveform with segment boundaries and streaming predictions over time.
    """
    import matplotlib.pyplot as plt

    segments_per_sample = length // timestep
    n_frames = preds_raw.shape[0]
    frame_duration_ms = (timestep / sample_rate) * 1000
    time_frames = np.arange(n_frames) * frame_duration_ms
    pred_class = np.argmax(preds_raw, axis=1)

    # Color each dot by correct (green) vs incorrect (red) to highlight prediction quality
    segment_idx_per_frame = np.arange(n_frames) // segments_per_sample
    true_labels_per_frame = np.array(labels_10)[np.minimum(segment_idx_per_frame, n_stream_samples - 1)]
    correct = (pred_class == true_labels_per_frame)
    dot_colors = np.where(correct, '#2d5a27', '#bdc3c7')  # green = correct, light grey = incorrect

    # Common x-axis for vertical alignment of both plots
    total_duration_ms = (n_stream_samples * length) / sample_rate * 1000

    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True, constrained_layout=True)
    t_audio_ms = np.arange(len(audio_1d)) / sample_rate * 1000
    axs[0].plot(t_audio_ms, audio_1d, color='#2d5a27', linewidth=0.5, alpha=0.9)
    for i in range(1, n_stream_samples):
        boundary_ms = (i * length) / sample_rate * 1000
        axs[0].axvline(boundary_ms, color='gray', linestyle='--', alpha=0.7)
    axs[0].set_ylabel('Amplitude')
    axs[0].set_title('Audio stream (this is how it comes in)')
    axs[0].set_xlim(0, total_duration_ms)

    axs[1].scatter(time_frames, pred_class, s=8, alpha=0.6, c=dot_colors)
    for i in range(1, n_stream_samples):
        boundary_ms = (i * length) / sample_rate * 1000
        axs[1].axvline(boundary_ms, color='gray', linestyle='--', alpha=0.7)
    for i in range(n_stream_samples):
        start_f = i * segments_per_sample
        end_f = (i + 1) * segments_per_sample
        mid = (start_f + end_f) // 2
        axs[1].annotate(
            label_names[segment_preds[i]],
            (time_frames[mid], pred_class[mid]),
            fontsize=8, ha='center'
        )
    axs[1].set_xlabel('Time (ms)')
    axs[1].set_ylabel('Predicted class')
    axs[1].set_title('Streaming predictions (how we predict in real time)')
    axs[1].set_yticks(range(len(label_names)))
    axs[1].set_yticklabels(label_names)
    axs[1].set_ylim(-0.5, len(label_names) - 0.5)
    axs[1].set_xlim(0, total_duration_ms)
    plt.show()


def plot_power_estimates(clocks_mhz, duty_power_uw):
    """Plot duty-cycle power vs clock."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(clocks_mhz, duty_power_uw, 's-', color='#3498db')
    ax.set_xlabel('Clock (MHz)')
    ax.set_ylabel('Duty-cycle power (μW)')
    ax.set_title('Average power (streaming use)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_duty_cycle_explanation(
    inference_clk,
    buffer_ms=16,
    clocks_mhz=(25, 100),
    dynamic_50mhz_uw=550,
    leakage_uw=14,
):
    """
    Plot power vs time for two clock speeds.

    Within one buffer period (16 ms): at lower clock the hardware stays active longer
    (same cycles ÷ lower freq = longer time) but draws less power; at higher clock it
    stays active shorter but draws more power. Area under each curve = same energy.

    Args:
        inference_clk: Hardware cycles per inference.
        buffer_ms: Buffer period in ms (e.g. 16 for KWS).
        clocks_mhz: Tuple of two clock speeds to compare (e.g. (25, 100)).
        dynamic_50mhz_uw: Dynamic power at 50 MHz (μW).
        leakage_uw: Leakage power (μW).
    """
    import matplotlib.pyplot as plt

    ref_mhz = 50
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    for idx, (ax, f_mhz) in enumerate(zip(axs, clocks_mhz)):
        f_hz = f_mhz * 1e6
        active_ms = (inference_clk / f_hz) * 1000
        idle_ms = max(0, buffer_ms - active_ms)
        scale = f_mhz / ref_mhz
        dynamic_uw = dynamic_50mhz_uw * scale
        total_uw = dynamic_uw + leakage_uw
        duty = min(active_ms / buffer_ms, 1.0)
        avg_power = total_uw * duty + leakage_uw * (1 - duty)

        # Step function: [0, t_active, t_active, buffer_ms], [P_active, P_active, P_idle, P_idle]
        t = [0, active_ms, active_ms, buffer_ms]
        p = [total_uw, total_uw, leakage_uw, leakage_uw]
        ax.step(t, p, where='post', color='#2d5a27', linewidth=2, label='Power')
        ax.fill_between(t, p, alpha=0.3, step='post', color='#2d5a27')

        ax.set_xlim(0, buffer_ms + 1)
        ax.set_ylim(0, total_uw * 1.15)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Power (μW)')
        ax.set_title(f'{f_mhz} MHz\n~{avg_power:.1f} μW avg (area = E)')
        ax.axvline(buffer_ms, color='#7f8c8d', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Power vs time: duty cycle — same area (energy) per inference', fontsize=11, y=1.02)
    _f_hz = clocks_mhz[0] * 1e6
    _active_ms = (inference_clk / _f_hz) * 1000
    _scale = clocks_mhz[0] / ref_mhz
    _total_uw = dynamic_50mhz_uw * _scale + leakage_uw
    _duty = min(_active_ms / buffer_ms, 1.0)
    _avg_uw = _total_uw * _duty + leakage_uw * (1 - _duty)
    _e_streaming = _avg_uw * (buffer_ms / 1000)
    _total_100 = dynamic_50mhz_uw * (clocks_mhz[1] / ref_mhz) + leakage_uw
    takeaway = (
        f"Takeaway: ~{_avg_uw:.0f} μW average power for always-on real-time KWS. "
        f"~{_e_streaming:.2f} μJ per inference (constant with clock). "
        f"Active power values (e.g. ~{_total_uw:.0f} μW at {clocks_mhz[0]} MHz, ~{_total_100:.0f} μW at {clocks_mhz[1]} MHz) "
        f"come from the power estimates table above."
    )
    fig.text(0.5, 0.02, textwrap.fill(takeaway, width=90), ha='center', va='bottom', fontsize=9,
             transform=fig.transFigure, color='#555555')
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show()


def plot_active_power_vs_clock(
    clocks_mhz,
    ref_clock_mhz=50,
    dynamic_50mhz_uw=550,
    leakage_uw=14,
):
    """
    Plot active (burst) power vs clock — without duty cycle.

    Shows raw power during inference. For streaming power (with duty cycle), see
    plot_duty_cycle_explanation.
    """
    import matplotlib.pyplot as plt

    active_power_uw = []
    for f_mhz in clocks_mhz:
        scale = f_mhz / ref_clock_mhz
        total_uw = dynamic_50mhz_uw * scale + leakage_uw
        active_power_uw.append(total_uw)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(clocks_mhz, active_power_uw, 'o-', color='#e74c3c', label='Active power (burst)')
    ax.set_xlabel('Clock (MHz)')
    ax.set_ylabel('Power (μW)')
    ax.set_title('Active power vs clock — without duty cycle')
    ax.legend()
    ax.grid(True, alpha=0.3)
    takeaway = (
        "Takeaway: Raw power during inference scales with clock. For streaming KWS power "
        "(with duty cycle), see the duty-cycle graph below."
    )
    fig.text(0.5, 0.02, textwrap.fill(takeaway, width=90), ha='center', va='bottom', fontsize=9,
             transform=fig.transFigure, color='#555555')
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.show()


def plot_latency_vs_clock(clocks_mhz, latency_ms, buffer_ms=16):
    """
    Plot inference latency vs clock with buffer headroom line.

    X: clock (MHz), Y: inference latency (ms).
    Horizontal line at BUFFER_MS shows real-time headroom (if latency approaches buffer, you miss real-time).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(clocks_mhz, latency_ms, 'o-', color='#2d5a27', label='Inference latency')
    ax.axhline(buffer_ms, color='#e74c3c', linestyle='--', alpha=0.8, label=f'Buffer ({buffer_ms} ms)')
    ax.set_xlabel('Clock (MHz)')
    ax.set_ylabel('Inference latency (ms)')
    ax.set_title('Latency vs Clock — speed/latency knob')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(clocks_mhz.min() - 2, clocks_mhz.max() + 2)
    takeaway = (
        f"Takeaway: Keep inference latency below the buffer line ({buffer_ms} ms) for real-time KWS. "
        "Higher clock speeds reduce latency and increase headroom; lower clock speeds risk "
        "missing the buffer and dropping frames."
    )
    fig.text(0.5, 0.02, textwrap.fill(takeaway, width=90), ha='center', va='bottom', fontsize=9,
             transform=fig.transFigure, color='#555555')
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show()


def plot_energy_per_inference_vs_clock(
    clocks_mhz,
    latency_ms,
    duty_power_uw,
    buffer_ms=16,
    ref_clock_mhz=50,
    dynamic_50mhz_uw=550,
    leakage_uw=14,
):
    """
    Plot energy per inference vs clock — two curves.

    E_active = P_active * t_inf (compute-only active energy)
    E_period = P_avg * T_buffer (streaming duty-cycle system energy per period)

    Explains compute cost vs always-on system cost.
    """
    import matplotlib.pyplot as plt

    active_energy_uj = []
    period_energy_uj = []
    for i, f_mhz in enumerate(clocks_mhz):
        scale = f_mhz / ref_clock_mhz
        total_uw = dynamic_50mhz_uw * scale + leakage_uw
        t_inf_s = latency_ms[i] / 1000
        e_active = total_uw * t_inf_s  # μW * s = μJ
        e_period = duty_power_uw[i] * (buffer_ms / 1000)  # μW * s = μJ
        active_energy_uj.append(e_active)
        period_energy_uj.append(e_period)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(clocks_mhz, active_energy_uj, 'o-', color='#e74c3c', label='Active (compute-only) E_active = P_active × t_inf')
    ax.plot(clocks_mhz, period_energy_uj, 's-', color='#3498db', label='System (streaming) E_period = P_avg × T_buffer')
    ax.set_xlabel('Clock (MHz)')
    ax.set_ylabel('Energy per inference (μJ)')
    ax.set_title('Energy per Inference vs Clock — active vs streaming')
    ax.legend()
    ax.grid(True, alpha=0.3)
    takeaway = (
        "Takeaway: Active (compute-only) energy drops at higher clock speeds because inference "
        "finishes sooner. Streaming (system) energy per inference stays roughly constant because "
        "the duty cycle adjusts: faster clock = shorter active time per buffer. For battery life, "
        "focus on the streaming curve."
    )
    fig.text(0.5, 0.02, textwrap.fill(takeaway, width=90), ha='center', va='bottom', fontsize=9,
             transform=fig.transFigure, color='#555555')
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.show()


def plot_power_vs_clock(
    clocks_mhz,
    latency_ms,
    duty_power_uw,
    buffer_ms=16,
    ref_clock_mhz=50,
    dynamic_50mhz_uw=550,
    leakage_uw=14,
):
    """
    Plot average power vs clock — two curves.

    Active power: dynamic + leakage during compute (burst power).
    Duty-cycle average power: streaming average (always-on average).
    """
    import matplotlib.pyplot as plt

    active_power_uw = []
    for f_mhz in clocks_mhz:
        scale = f_mhz / ref_clock_mhz
        total_uw = dynamic_50mhz_uw * scale + leakage_uw
        active_power_uw.append(total_uw)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(clocks_mhz, active_power_uw, 'o-', color='#e74c3c', label='Active power (burst)')
    ax.plot(clocks_mhz, duty_power_uw, 's-', color='#3498db', label='Duty-cycle average (streaming)')
    ax.set_xlabel('Clock (MHz)')
    ax.set_ylabel('Power (μW)')
    ax.set_title('Average Power vs Clock — burst vs always-on average')
    ax.legend()
    ax.grid(True, alpha=0.3)
    takeaway = (
        "Takeaway: Active power scales with clock speed during inference bursts. Duty-cycle "
        "average power (streaming) stays flat because the device spends less time active at "
        "higher clocks. For always-on KWS, the duty-cycle average is the relevant metric."
    )
    fig.text(0.5, 0.02, textwrap.fill(takeaway, width=90), ha='center', va='bottom', fontsize=9,
             transform=fig.transFigure, color='#555555')
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.show()


def plot_throughput_vs_clock(clocks_mhz, inference_clk, buffer_ms=16):
    """
    Plot max FPS vs clock with required FPS line.

    Y: max_fps = f_hz / inference_cycles
    Horizontal line: required_fps = 1000 / BUFFER_MS (e.g. 62.5 fps for 16 ms buffer)

    Shows whether the system can keep up.
    """
    import matplotlib.pyplot as plt

    max_fps = []
    for f_mhz in clocks_mhz:
        f_hz = f_mhz * 1e6
        max_fps.append(f_hz / inference_clk)

    required_fps = 1000 / buffer_ms

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(clocks_mhz, max_fps, 'o-', color='#2d5a27', label='Max FPS')
    ax.axhline(required_fps, color='#e74c3c', linestyle='--', alpha=0.8, label=f'Required FPS ({required_fps:.1f})')
    ax.set_xlabel('Clock (MHz)')
    ax.set_ylabel('Throughput (FPS)')
    ax.set_title('Throughput (Max FPS) vs Clock — can it keep up?')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(clocks_mhz.min() - 2, clocks_mhz.max() + 2)
    takeaway = (
        f"Takeaway: Stay above the required FPS line ({required_fps:.1f} fps for {buffer_ms} ms buffer) "
        "for real-time streaming. Higher clock speeds increase max FPS and headroom; below the line "
        "the system cannot keep up with the audio stream."
    )
    fig.text(0.5, 0.02, textwrap.fill(takeaway, width=90), ha='center', va='bottom', fontsize=9,
             transform=fig.transFigure, color='#555555')
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.show()


def build_stateful_eval_dataset(X_test, y_test, length=16384, timestep=256):
    """
    Build a tf.data.Dataset for frame-by-frame stateful evaluation.

    Yields one (frame, label) per 256-sample chunk in order: clip 0 frames 0..63,
    then clip 1 frames 0..63, etc. Each frame has shape (1, 256, 1), label (1,).

    Returns:
        val_ds: tf.data.Dataset
        total_steps: int (N * 64)
        segments_per_sample: int (64)
    """
    import tensorflow as tf

    N = X_test.shape[0]
    if X_test.ndim == 2:
        X_test = X_test[..., np.newaxis]
    segments_per_sample = length // timestep
    # (N, 64, 256, 1) -> (N*64, 256, 1)
    frames = np.reshape(X_test, (N, segments_per_sample, timestep, -1)).reshape(
        N * segments_per_sample, timestep, -1
    )
    labels = np.repeat(np.asarray(y_test, dtype=np.int64)[:N], segments_per_sample)

    ds = tf.data.Dataset.from_tensor_slices((frames.astype(np.float32), labels))

    def add_batch_dim(f, l):
        return tf.reshape(f, (1, timestep, -1)), tf.reshape(l, (1,))

    val_ds = ds.map(add_batch_dim)
    total_steps = N * segments_per_sample
    return val_ds, total_steps, segments_per_sample


def evaluate_stateful_model(
    model,
    val_ds,
    total_steps,
    segments_per_sample,
    in_akida=False,
    smooth_window=0,
    response_frames=None,
):
    """
    Evaluate the stateful model frame-by-frame (one clip at a time, no batching over clips).

    Segment prediction = aggregate frame logits -> one vector -> argmax. Two knobs:
    - response_frames (N): use only first N frame logits (fast response; less latency).
    - smooth_window: temporally smooth frame logits with sliding mean before the final mean.

    Args:
        model: Stateful Keras model or Akida model (already converted/mapped).
        val_ds: tf.data.Dataset yielding (frame (1, 256, 1), label (1,)) per step.
        total_steps: Total number of steps (N * 64).
        segments_per_sample: Steps per clip (64).
        in_akida: True if model is akida.Model.
        smooth_window: If > 0, apply sliding-window smoothing over the (64 or first N)
            frame logits, then mean and argmax. 0 = no smoothing.
        response_frames: If set (e.g. 16, 32), use only the first this many frame logits
            for the segment decision (fast response). None = use all 64 frames.
    """
    import tensorflow as tf
    from tqdm import tqdm

    if in_akida:
        model_func = None
    else:
        model_func = tf.function(model)

    use_raw_list = smooth_window > 0 or response_frames is not None
    if use_raw_list:
        raw_logits_list = []
    else:
        cumulated_preds = None

    correct = 0
    num_samples = 0
    pbar = tqdm(total=total_steps, desc="Eval")

    for batch_id, (frame, label) in enumerate(val_ds):
        if in_akida:
            frame_np = frame.numpy().astype(np.float32)  # (1, 256, 1)
            # Match Section 7: (batch, 1, 256, 1) int16 via direct cast (no 32767 scaling)
            frame_in = np.expand_dims(frame_np, axis=1).astype(np.int16)
            prediction = model.forward(frame_in)
        else:
            prediction = model_func(frame)

        # Per-frame logits: match Section 7 — squeeze, then ensure (1, n_classes)
        pred_arr = prediction.numpy() if hasattr(prediction, "numpy") else np.asarray(prediction)
        out = np.squeeze(pred_arr)
        if out.ndim == 1:
            out = out[np.newaxis, :]
        pred_frame = out  # (1, n_classes)

        reset = (batch_id % segments_per_sample) == (segments_per_sample - 1)

        if use_raw_list:
            raw_logits_list.append(np.squeeze(pred_frame))
            if reset:
                raw_logits = np.stack(raw_logits_list, axis=0)  # (64, n_classes)
                n_use = min(int(response_frames), raw_logits.shape[0]) if response_frames is not None else raw_logits.shape[0]
                logits_for_seg = raw_logits[:n_use]
                if smooth_window > 0:
                    smoothed = sliding_mean(logits_for_seg, window_size=min(smooth_window, n_use))
                    seg_logits = smoothed.mean(axis=0)
                else:
                    seg_logits = logits_for_seg.mean(axis=0)
                pred_class = np.argmax(seg_logits)
                lab = label.numpy() if hasattr(label, "numpy") else np.asarray(label)
                correct += 1 if pred_class == np.squeeze(lab).item() else 0
                num_samples += 1
                raw_logits_list = []
        else:
            # Section 7–matching baseline: mean of raw logits over 64 frames then argmax
            if batch_id % segments_per_sample == 0:
                cumulated_preds = np.asarray(pred_frame, dtype=np.float64)
            else:
                cumulated_preds = cumulated_preds + pred_frame
            if reset:
                pred_class = np.argmax(cumulated_preds, axis=-1).item()
                lab = label.numpy() if hasattr(label, "numpy") else np.asarray(label)
                correct += 1 if pred_class == np.squeeze(lab).item() else 0
                num_samples += 1
                cumulated_preds = None

        if reset:
            if in_akida:
                try:
                    import akida
                    model = akida.Model(model.layers)
                except Exception:
                    pass
            else:
                model.reset_states()
        pbar.update(1)

    pbar.close()
    accuracy = correct / num_samples * 100
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy


def evaluate_stateful_full(
    X_test,
    y_test,
    model,
    length=16384,
    timestep=256,
    in_akida=None,
    smooth_window=0,
    response_frames=None,
    verbose=1,
):
    """
    Evaluate the stateful model on the full test set (frame-by-frame, one clip at a time).

    Builds the dataset, runs evaluate_stateful_model, returns accuracy.

    Args:
        in_akida: If None (default), auto-detect from model type. Set True/False to override.
        smooth_window: If > 0, apply sliding-window smoothing to (64 or first N) frame
            logits, then mean and argmax. 0 = no smoothing.
        response_frames: If set (e.g. 16, 32), use only first N frame logits per segment
            (fast response). None = use all 64 frames.
    """
    try:
        import akida
        _in_akida = isinstance(model, akida.Model)
    except ImportError:
        _in_akida = False
    if in_akida is not None:
        _in_akida = in_akida

    val_ds, total_steps, segments_per_sample = build_stateful_eval_dataset(
        X_test, y_test, length=length, timestep=timestep
    )
    if verbose:
        N = X_test.shape[0]
        print(f"Evaluating stateful model on {N} clips ({total_steps} steps)...")
        if response_frames is not None:
            if response_frames >= segments_per_sample:
                print(f"Using full segment ({segments_per_sample} frames).", end="")
            else:
                print(f"Fast response: first {response_frames} frame logits per segment.", end="")
            if smooth_window > 0:
                print(f" Sliding window smoothing: {smooth_window}.")
            else:
                print()
        elif smooth_window > 0:
            print(f"Sliding-window smoothing (window={smooth_window}) on raw logits, then mean over segment.")
    return evaluate_stateful_model(
        model,
        val_ds,
        total_steps,
        segments_per_sample,
        in_akida=_in_akida,
        smooth_window=smooth_window,
        response_frames=response_frames,
    )




def preprocess_sc10(num_test=500, length=16384, target_sample_rate=16000):
    """
    [Optional] Load Speech Commands v0.02 from TensorFlow Datasets and return
    test waveforms and labels in the same format as the .npy files.
    Requires tensorflow_datasets and network access on first run.
    Returns X_test (n, length, 1), y_test (n,) with SC10 class indices 0–9.
    """
    import tensorflow_datasets as tfds
    from tqdm import tqdm

    builder = tfds.builder('speech_commands')
    builder.download_and_prepare()
    SC10 = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
    label_to_idx = {lbl: i for i, lbl in enumerate(SC10)}
    ds = builder.as_dataset(split='test')
    X_list, y_list = [], []
    for ex in tqdm(tfds.as_numpy(ds), total=num_test, desc='Loading SC10'):
        if len(X_list) >= num_test:
            break
        label = ex['label'].item()
        keyword = builder.info.features['label'].names[label]
        if keyword not in label_to_idx:
            continue
        wav = ex['audio'].astype(np.float32) / 32768.0
        if len(wav) != target_sample_rate:
            continue
        if len(wav) < length:
            wav = np.pad(wav, (0, length - len(wav)))
        else:
            wav = wav[:length]
        X_list.append(wav)
        y_list.append(label_to_idx[keyword])
    X_test = np.stack(X_list)[..., np.newaxis]
    y_test = np.array(y_list)
    return X_test, y_test
