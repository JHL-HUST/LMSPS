# LDMLP on HGB datasets

## Search stage

```bash
python train_search.py --stage 50 --dataset DBLP --num-hops 5 --residual --amp --seeds 42 --ns 20 --all_path --gpu 0

python train_search.py --stage 50 --dataset IMDB --num-hops 4 --amp --seeds 42 --ns 20 --all_path --gpu 0

python train_search.py --stage 50 --dataset ACM --num-hops 5 --amp --seeds 1 --ns 20 --all_path --gpu 0
```

## Training stage

After search stage, you can add you search result in `arch.py`, and change the args `arch`.

```bash
seeds='1 2 3 4 5 6 7 8 9 10'

python main_path.py --stage 200 --dataset DBLP --n-layers-2 3 --num-hops 5 --hidden 512 --embed-size 512 --residual --amp --seeds $seeds --arch dblp_hop5_top20 --gpu 0

python main_path.py --stage 200 --dataset ACM --n-layers-2 3 --num-hops 5 --hidden 512 --embed-size 512 --amp --seeds $seeds --arch acm_hop5_top20 --gpu 0

python main_path.py --stage 200 --dataset IMDB --n-layers-2 4 --num-hops 4 --hidden 512 --embed-size 512 --amp --seeds $seeds --arch imdb_hop4_top20 --gpu 0
```

