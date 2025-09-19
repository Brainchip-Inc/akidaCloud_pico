Overview
--------

This repository provides examples for deploying machine learning tasks, such as image classification and object detection, on the Akida 2 FPGA cloud platform.

Setup
-----

1.  Install the necessary dependencies:
    
        conda install -c conda-forge jupyterlab
        pip install -r requirements.txt
    
3.  Set up models and datasets:
    
        bash get_models.sh
    
4. Start JupyterLab:

        ./examples/start-jupyterlab.sh
