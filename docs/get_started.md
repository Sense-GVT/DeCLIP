## Get Started

Install PyTorch. The code has been tested with CUDA 11.2/CuDNN 8.1.0, PyTorch 1.8.1.


First, prepare pre-training datasets and downstream classification datasets through [get_started.md](docs/get_started.md#installation). 

We organize the different models trained on different data through separate [experimental catalogs] (experiments/), you can check the dir for detail.

#### 1. Pre-training

You can run `run.sh` directly to train the corresponding model. We train most of our models on 4x8-gpu nodes. Check the config in the experiment directory of the corresponding model for details.

#### 2. Zero-shot Evalution

You can add a argument `--evaluate` on run script for zero-shot evalution.
