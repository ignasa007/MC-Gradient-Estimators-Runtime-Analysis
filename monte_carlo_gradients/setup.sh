#!/bin/bash

conda create -n mc_grads_venv python=3.6
conda activate mc_grads_venv

conda install pip
pip install -r monte_carlo_gradients/requirements.txt