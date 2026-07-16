Overview
--------

This repository provides examples for deploying machine learning tasks, such as key word spotting, on the Akida Pico FPGA cloud platform.

Setup
-----

1.  Install the necessary dependencies:

        # System binaries (ffmpeg is required by tensorflow_datasets for the Speech Commands dataset)
        conda install -c conda-forge jupyterlab ffmpeg
        pip install -r requirements.txt

2.  Start JupyterLab:

        ./examples/start-jupyterlab.sh

Model weights and calibration data needed by the example notebooks are bundled in
the repository under the corresponding example folders (e.g.
`examples/kws/weights/` and `examples/kws/calibration/`) — no separate
download step is required. Example datasets are fetched on first run via
`tensorflow_datasets`.
