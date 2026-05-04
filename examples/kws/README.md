# KWS Demo: 12-Class SSM Keyword Spotting on Pico

This demo shows how a **State Space Model (SSM)** keyword-spotting model runs on **Pico**, BrainChip's FPGA hardware, for low-power edge inference. The model classifies 12 categories: 10 keywords (down, go, left, no, off, on, right, stop, up, yes) plus **silence** and **unknown**.

## Folder structure

```
kws/
├── kws_sc12.ipynb                          # Main notebook (12-class pipeline)
├── kws_utils.py                            # Helper functions
├── README.md
├── weights/
│   └── tenn_recurrent_sc12.h5              # Pre-trained 12-class weights
└── calibration/
    └── sc10_batch100_1024samples.npz       # Quantization calibration data
```

| Path | Purpose |
|------|---------|
| `kws_sc12.ipynb` | Main notebook: data → model → stateful conversion → quantization → Pico mapping → power estimates → streaming inference |
| `kws_utils.py` | Helper functions for data loading, streaming inference, evaluation, and plotting |
| `weights/tenn_recurrent_sc12.h5` | Pre-trained weights for the 12-class SSM model |
| `calibration/sc10_batch100_1024samples.npz` | Representative samples used by `quantizeml` to calibrate int8/int16 scales |

## Quick start

1. **Prerequisites:** Python with `akida`, `akida_models`, `quantizeml`, `cnn2snn`, `tensorflow`, `tensorflow_datasets`, `numpy`, `matplotlib`.
2. **Pico:** Connect Pico via USB. Check with `akida devices`.
3. **Run:** Open `kws_sc12.ipynb` and run all cells in order (or **Run All**).

## Test data

The notebook loads the SC12 test set automatically from **TensorFlow Datasets** (`tfds.builder("speech_commands")`). On the first run, TFDS downloads the dataset (~2 GB) and caches it locally to `~/tensorflow_datasets/`. Subsequent runs use the cache — no internet required after the first download.

No `X_test.npy` / `y_test.npy` files are needed.

## What `kws_utils.py` provides (functions used by the notebook)

- **Data loading:** `load_sc12_test_data()` — loads the 12-class test set from TFDS
- **Evaluation:** `evaluate_stateful_full()` — stateful accuracy on the full test set running on the Akida-mapped model (i.e. on the Pico FPGA)
- **Streaming helpers:** `sliding_mean`, `softmax` — used inside the notebook's `generate_predictions()` for frame-level smoothing and probability normalization
- **Plots:**
  - `plot_power_estimates` — duty-cycle power vs clock
  - `plot_duty_cycle_explanation` — single-buffer-period view at two clock speeds
  - `plot_throughput_vs_clock` — max FPS vs clock with required-FPS line
  - `plot_streaming_demo` — waveform + segment-level streaming predictions
  - `plot_streaming_debug` — raw logits and smoothed logits over time
- **Constants:** `LABEL_NAMES` (12 class names, indices 0–11), `SAMPLE_RATE` (16000)

## Pipeline overview

1. **Load test data** from TFDS (12 classes, 16-kHz, 1-second clips)
2. **Load model and weights** — `akida_models.tenn_recurrent_sc10(num_classes=12)` + `weights/tenn_recurrent_sc12.h5`
3. **Stateful conversion** — rewrite the model to consume 256-sample chunks while carrying state
4. **Quantize** to int8 weights / int8 activations / int16 input using calibration data
5. **Convert to Akida** and **map** to the Pico device
6. **Performance**: latency, FPS, and duty-cycle power estimates across 25–100 MHz
7. **Stateful evaluation** on the full SC12 test set running on the Pico FPGA
8. **Streaming inference demo** — concatenate clips into a stream, run frame-by-frame, plot waveform + predictions

Run cells in order; later sections depend on earlier ones (e.g. power estimates use `m` from the latency cell; the streaming demo and evaluation require `model_akida` from the mapping cell).
