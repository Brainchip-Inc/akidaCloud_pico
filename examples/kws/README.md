# KWS Demo: SSM Keyword Spotting on Pico

This demo shows how a **State Space Model (SSM)** keyword-spotting model runs on **Pico**, BrainChip's FPGA hardware, for low-power edge inference.

## What's in this folder

| File | Purpose |
|------|---------|
| **kws.ipynb** | Main notebook: model loading → stateful conversion → quantization → Pico mapping → power estimates → streaming inference |
| **kws_utils.py** | Helper functions for data loading, streaming inference, and plotting (used by the notebook) |

## Quick start

1. **Prerequisites:** Python with `akida`, `akida_models`, `quantizeml`, `cnn2snn`, TensorFlow, numpy, matplotlib.
2. **Pico:** Connect Pico via USB. Check with `akida devices`.
3. **Run:** Open `kws.ipynb` and run all cells in order (or use **Run All**).

## Required files

Place these in the **same folder** as the notebook:

| File | Description |
|------|-------------|
| `X_test.npy` | Test audio waveforms (shape: n_samples × 16384 × 1) |
| `y_test.npy` | Test labels (0–9 for SC10 keywords) |
| `tenn_recurrent_sc10.h5` | Pre-trained model weights |
| `sc10_batch100_1024samples.npz` | Calibration data for quantization (optional; random data used if missing) |

## What kws_utils.py provides

- **Data:** `load_test_data()` — loads `X_test.npy` and `y_test.npy`
- **Streaming:** `sliding_mean`, `softmax` — used for frame-level smoothing
- **Evaluation:** `evaluate_stateful_full()` — stateful accuracy on the full test set
- **Plots:** `plot_streaming_demo`, `plot_streaming_debug` — audio + predictions; `plot_power_estimates`, `plot_duty_cycle_explanation`, `plot_throughput_vs_clock` — power and throughput

## Pipeline overview

1. Load test data and model weights
2. Convert model to **stateful** form (256-sample chunks for streaming)
3. **Quantize** to int8/int16
4. **Map** to Akida/Pico
5. View **power estimates** and **throughput** vs clock
6. Run **streaming inference** on device

Run cells in order; later sections depend on earlier ones (e.g. power estimates use `m` from the latency cell).
