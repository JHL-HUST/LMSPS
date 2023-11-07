# LMSPS on Ogbn-mag

## Search stage

To save processing time, you can download the label propagation features in https://drive.google.com/file/d/1h8FsEd1dONlx0lHxEGm-Y68DvLfo34AM/view?usp=drive_link.

```bash
python train_search.py  --label-feats --residual --label_residual  --amp  --all_path
```

## Training stage

After search stage, you can add you search result in `arch.py`, and change the args `arch`.

```bash
python main_path.py   --label-feats   --residual --bns --label-bns   --amp  --arch ogbn_withLabel
```