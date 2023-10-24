# LMSPS on HGB datasets

## Search stage

```bash
python train_search.py --stage 200 --dataset DBLP --num-hops 6 --residual --amp --seeds 0 --ns 30 --num_final 60 --all_path --repeat 200 --gpu 0

python train_search.py --stage 200 --dataset IMDB --num-hops 5 --amp --seeds 0 --ns 30 --num_final 60 --all_path --repeat 200 --gpu 0

python train_search.py --stage 200 --dataset ACM --num-hops 5 --amp --seeds 0 --ns 30 --num_final 60 --all_path --repeat 200 --gpu 0
```

## Training stage

After search stage, you can add you search result in `arch.py`, and change the args `arch`.

```bash
seeds='1 2 3 4 5 6 7 8 9 10'

python main_path.py --stage 200 --dataset DBLP --n-layers-2 3 --num-hops 6 --hidden 512 --embed-size 512 --residual --amp --seeds $seeds --arch dblp --gpu 0

python main_path.py --stage 200 --dataset IMDB --n-layers-2 4 --num-hops 5 --hidden 512 --embed-size 512 --amp --seeds $seeds --arch imdb --gpu 0

python main_path.py --stage 200 --dataset ACM --n-layers-2 3 --num-hops 5 --hidden 512 --embed-size 512 --amp --seeds $seeds --arch acm --gpu 0

```