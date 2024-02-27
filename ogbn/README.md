# LMSPS on Ogbn-mag

To save processing time, you can download the label propagation features in https://drive.google.com/file/d/1h8FsEd1dONlx0lHxEGm-Y68DvLfo34AM/view?usp=drive_link.



## Search stage

```bash
python train_search.py  --label-feats --residual --label_residual  --amp  --all_path
```



## Training stage

After search stage, you can add you search result in `arch.py`, and change the `args.arch`.

You can also **directly run the following command** to get the results in our paper with our searched meta-paths. 




Directly run LMSPS without any enhancement:
```bash
python main_path.py  --stage 200   --residual --bns --label-bns   --amp  --arch ogbn_withoutLabel
```
After you finish the training, you should get the test accuracy of 55.02%.



Run LMSPS with label aggregation and multi-stage learning:
```bash
python main_path.py   --label-feats   --residual --bns --label-bns   --amp  --arch ogbn_withLabel
```
After you finish the training, you should get the test accuracy of 57.97%.