![BrainChip](examples/img/BC-banner-1920x200.jpg)

# Akida Pico FPGA — Cloud Examples

Runnable examples for deploying neural-network models to **BrainChip Akida Pico** — an ultra-low-power,
event-based neural IP for always-on 1-D sensing — on the **Akida Pico FPGA cloud platform** (a single-NP
Pico IP implemented on a Xilinx FPGA and accessed through JupyterLab).

Each example takes a trained model the full path to on-device inference:

> **build → convert to a streaming (stateful) form → quantize to int8 → convert with Akida's MetaTF
> toolchain → map onto the Pico → measure and run on hardware.**

The Akida software toolchain — **MetaTF** (`akida`, `cnn2snn`, `quantizeml`, `akida_models`) — and full
API documentation is here: **https://doc.brainchipinc.com/index.html**

## Examples

| Example | Task | What it demonstrates |
|---------|------|----------------------|
| [Keyword Spotting](examples/kws/) | 12-class speech-command recognition (audio, 16 kHz) | Streaming SSM keyword spotting on Pico — data → stateful conversion → quantization → Akida mapping → latency/throughput/power → streaming inference. |
| [Bearing Fault Detection](examples/fault_detection/) | Multi-label vibration fault detection (accelerometer, 42 kHz) | The same pipeline on a 1-D vibration stream — real-time multi-label fault detection, hardware metrics, a float-vs-Akida comparison, and an accuracy-vs-decision-latency study. |

Each example folder has its own `README.md` with the details.

## Setup

The Akida Cloud host already has the Pico FPGA attached and conda/Python available.

1. **Install dependencies:**

   ```bash
   conda install -c conda-forge jupyterlab ffmpeg
   pip install -r requirements.txt
   ```

   (`ffmpeg` is used by `tensorflow_datasets` to prepare the Speech Commands dataset for the Keyword
   Spotting example.)

2. **Launch JupyterLab:**

   ```bash
   ./start-jupyterlab.sh
   ```

3. **Open an example** — e.g. `examples/kws/kws_sc12.ipynb` or
   `examples/fault_detection/fault_detection_inference.ipynb` — and run all cells in order.

## The Pico device

The examples run on a real Akida Pico device. Confirm it is visible before running:

```bash
akida devices            # or:  python -c "import akida; print(akida.devices())"
```

You should see one device. Hardware and platform details are in
[examples/Akida_Cloud_Specs.md](examples/Akida_Cloud_Specs.md).
