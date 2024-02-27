# LMSPS

This repository contains a PyTorch implementation of the preprint paper:  [Long-range Meta-path Search through Progressive Sampling on Large-scale Heterogeneous Information Networks](https://arxiv.org/abs/2307.08430v1)

## Requirements

All experiments are conducted on Tesla V100 GPU (16GB). 

Please install the requirements using the following command. (The python version is 3.9.18)

```setup
pip install -r requirements.txt
git clone https://github.com/Yangxc13/sparse_tools.git --depth=1
cd sparse_tools
python setup.py develop
cd ..
```

## Data preparation

For HGB datasets, you can download from [the source of HGB benchmark](https://github.com/THUDM/HGB), and extract content from these compresesed files under the folder `'./data/'`. Or you can run the script `download_hgb_datasets.sh`.

For experiments on the large dataset ogbn-mag, the dataset will be automatically downloaded from OGB challenge.

## Acknowledgment

This repository benefit from [SeHGNN](https://github.com/ICT-GIMLab/SeHGNN/tree/master/ogbn).
