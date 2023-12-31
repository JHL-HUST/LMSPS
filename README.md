# LMSPS

This repository contains a PyTorch implementation of the preprint paper:  [Long-range Meta-path Search through Progressive Sampling on Large-scale Heterogeneous Information Networks](https://arxiv.org/abs/2307.08430v1)

## Requirements

#### 1. Neural network libraries for GNNs

* [pytorch](https://pytorch.org/get-started/locally/)
* [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
* [dgl](https://www.dgl.ai/pages/start.html)

Please check your cuda version first and install the above libraries matching your cuda. If possible, we recommend to install the latest versions of these libraries.

#### 2. Other dependencies

Install other requirements:

```setup
pip install -r requirements.txt
git clone https://github.com/Yangxc13/sparse_tools.git --depth=1
cd sparse_tools
python setup.py develop
cd ..
```

## Data preparation

For HGB datasets, you can download from [the source of HGB benchmark](https://github.com/THUDM/HGB), and extract content from these compresesed files under the folder `'./data/'`.

For experiments on the large dataset ogbn-mag, the dataset will be automatically downloaded from OGB challenge.

## Acknowledgment
This repository benefit from [SeHGNN](https://github.com/ICT-GIMLab/SeHGNN/tree/master/ogbn).