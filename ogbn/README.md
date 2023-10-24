# LMSPS on Ogbn-mag

## Search stage

To save processing time, you can download the label propagation features in https://drive.google.com/file/d/1h8FsEd1dONlx0lHxEGm-Y68DvLfo34AM/view?usp=drive_link.

```bash
python train_search.py --stages 200 --num-hops 6 --label-feats --num-label-hops 4 --hidden 512 --residual --label_residual --lr 0.001 --weight-decay 0  --patience 100  --amp --seeds 42 --ns 30 --all_path  --repeat 200  --num_final 60 --gpu 0 
```

## Training stage

After search stage, you can add you search result in `arch.py`, and change the args `arch`.

```bash
python main_path.py --stages 200 200 200 200 200 200 --num-hops 6 --label-feats --num-label-hops 4 --n-layers-2 2 --n-layers-3 2 --residual --bns --label-bns --lr 3e-3 --weight-decay 0 --threshold 0.6 --patience 100 --gama 5 --amp --seeds 1 --gpu 0 --arch ogbn_withLabel
```