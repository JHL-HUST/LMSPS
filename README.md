# LMSPS

This repository contains a PyTorch implementation of the preprint paper:  [Long-range Meta-path Search on Large-scale Heterogeneous Graphs](https://arxiv.org/pdf/2307.08430.pdf).

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

## Run LMSPS

You can run LMSPS on HGB and ogbn-mag based on the command in [hgb](https://github.com/JHL-HUST/LMSPS/tree/main/hgb) and [ogbn-mag](https://github.com/JHL-HUST/LMSPS/tree/main/ogbn), respectively.

## Cite

If you use LMSPS in a scientific publication, we would appreciate citations to the following paper:

```
@misc{li2024longrange,
      title={Long-range Meta-path Search on Large-scale Heterogeneous Graphs}, 
      author={Chao Li and Zijie Guo and Qiuting He and Hao Xu and Kun He},
      year={2024},
      eprint={2307.08430},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```

## Acknowledgment

This repository benefit from [SeHGNN](https://github.com/ICT-GIMLab/SeHGNN/tree/master/ogbn).
