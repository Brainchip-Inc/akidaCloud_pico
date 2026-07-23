# Bearing Fault Detection with SSM on Pico (UORED-VAFCLS)

This demo shows how a **State Space Model (SSM)** bearing-fault model runs on **Pico**, BrainChip's FPGA hardware, for low-power always-on condition monitoring. From a single accelerometer channel it predicts **four independent fault types** — inner race, outer race, ball, and cage — as a **multi-label** output (a bearing can have more than one fault at once).

**Why Pico for this:** Akida Pico is BrainChip's ultra-low-power, event-based neural IP for always-on 1D sensing — and it is **sensor-agnostic**, so the same streaming TENNs / SSM engine used for audio keyword spotting runs vibration fault detection here. It reads raw int16 accelerometer samples directly (**no DSP, FFT, or spectrogram**), processing one small chunk at a time with a constant memory footprint. Paired with an ultra-low-power MCU it sits **always-on at the sensor** in the µW range while the host SoC stays in deep sleep, raising a single wake event only when a fault is detected — real-time predictive maintenance on-device, without the cloud, to reduce unplanned downtime.

## Folder structure

```
fault_detection/
├── fault_detection_inference.ipynb   # Main notebook (full pipeline)
├── fd_utils.py                       # Helper functions
├── README.md
├── weights/
│   └── tenn_recurrent_uored.h5       # Pre-trained kernelized weights
├── calibration/
│   └── uored_calibration.npz         # Quantization calibration data (256, 8192, 1)
└── test_data/
    ├── X_test.npy                    # Held-out test segments  (240, 41984, 1) int16
    ├── y_test.npy                    # Multi-label ground truth (240, 4) float32
    └── split_000.json                # Bearing-wise train/test split
```

| Path | Purpose |
|------|---------|
| `fault_detection_inference.ipynb` | Main notebook: data → model → stateful conversion → quantization → Pico mapping → hardware metrics → evaluation → streaming → decision-latency |
| `fd_utils.py` | Helpers for data loading, streaming inference, evaluation, decision-latency, and plotting |
| `weights/tenn_recurrent_uored.h5` | Pre-trained weights for the kernelized SSM model (`akida_models.tenn_recurrent_uored`) |
| `test_data/X_test.npy` / `test_data/y_test.npy` | Pre-extracted held-out test set (8 bearings, ~1-second segments) and its multi-label targets |
| `calibration/uored_calibration.npz` | Representative training-split samples used by `quantizeml` to calibrate int8/int16 scales |
| `test_data/split_000.json` | The bearing-wise train/test split (which bearing IDs are held out) |
| `uored_full.h5` *(not included)* | Raw 42 kHz UORED-VAFCLS recordings (~96 MB). **Not bundled** — it isn't needed to run the notebook, only to regenerate the arrays. Obtain it from the UORED-VAFCLS dataset and use `fd_utils.load_test_data` / `load_calibration_data`. |

## Quick start

1. **Prerequisites:**
    - Python packages: `akida`, `akida_models`, `quantizeml`, `cnn2snn`, `tensorflow`, `tf_keras`, `numpy`, `scikit-learn`, `matplotlib`, `h5py`.
    - No dataset download is required — the test and calibration arrays are bundled.
2. **Pico:** The Pico FPGA is already attached to the host. Verify it is visible with `akida devices` (or `import akida; akida.devices()` inside Python).
3. **Run:** Open `fault_detection_inference.ipynb` and run all cells in order (or **Run All**).

## Test data

The notebook loads a **pre-extracted** test set from `test_data/X_test.npy` / `test_data/y_test.npy` — no download or preprocessing step is needed. These are ~1-second (41984-sample) int16 accelerometer segments from **8 held-out bearings** (2 per fault type), selected by the bearing-wise split in `test_data/split_000.json` so that no bearing in the test set is seen during training.

The arrays were generated from the raw UORED-VAFCLS recordings (`uored_full.h5`, ~96 MB), which is **not bundled** in this folder. To regenerate them (e.g. for a different split or segment length), obtain `uored_full.h5` from the UORED-VAFCLS dataset and use `fd_utils.load_test_data(...)` / `fd_utils.load_calibration_data(...)`.

## What `fd_utils.py` provides (functions used by the notebook)

- **Data loading:** `load_test_data`, `load_calibration_data` — extract int16 segments and multi-label targets for a bearing-wise split from `uored_full.h5`
- **Evaluation:** `evaluate_stateful_multilabel` — frame-by-frame stateful evaluation (float or Akida) returning AUROC, F1, exact-match, Hamming, and per-fault AUROC; `print_metrics_report` formats it, and `print_param_summary` reports parameters/memory across the pipeline stages
- **Streaming helpers:** `run_streaming_inference`, `segment_predictions`, `sliding_mean`, `sigmoid` — chunk-by-chunk inference and per-segment aggregation
- **Decision latency:** `sweep_decision_latency`, `plot_accuracy_vs_latency` — accuracy as a function of how many chunks are accumulated before deciding
- **Plots:**
  - `plot_healthy_vs_faulty`, `plot_label_distribution` — healthy-vs-faulty waveforms and class balance
  - `plot_power_estimates` — duty-cycle power vs clock
  - `plot_duty_cycle_explanation` — single-buffer-period view at two clock speeds
  - `plot_latency_vs_clock`, `plot_throughput_vs_clock` — latency and max FPS vs clock
  - `plot_streaming_demo`, `plot_streaming_debug` — streaming predictions over time
  - `plot_evaluation_comparison`, `plot_per_fault_auroc` — metric comparisons
- **Constants:** `FAULT_NAMES` (`inner`, `outer`, `ball`, `cage`), `SAMPLE_RATE` (42000)

## Pipeline overview

1. **Load test data** — pre-extracted ~1-second int16 segments from held-out bearings
2. **Create model and load weights** — `akida_models.tenn_recurrent_uored(input_shape=(8192, 1))` + `weights/tenn_recurrent_uored.h5`
3. **Stateful conversion** — rewrite the model to consume 256-sample chunks while carrying state
4. **Quantize** to int8 weights / int8 activations / int16 input using the calibration data
5. **Convert to Akida** and **map** to the Pico device
6. **Performance** — latency, FPS, and duty-cycle power estimates across 25–100 MHz
7. **Evaluate** on the full held-out test set (float vs Akida) with multi-label metrics
8. **Streaming inference demo** — concatenate segments into a stream, run frame-by-frame, plot predictions
9. **Decision latency** — accuracy vs. number of accumulated chunks, to choose a prediction window

Run cells in order; later sections depend on earlier ones (e.g. the power estimates use `m` from the latency cell; the evaluation, streaming demo, and decision-latency sweep require `model_akida` from the mapping cell).

The model is trained on 8192-sample (~195 ms) windows and benchmarked at ~1-second segments; expect a macro-AUROC around **0.94** on the held-out bearings.
