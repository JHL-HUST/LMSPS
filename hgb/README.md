# LMSPS on HGB datasets



## Search stage

```bash
python train_search.py  --dataset DBLP --num-hops 6 --residual --amp  --all_path 

python train_search.py  --dataset IMDB --num-hops 5 --amp   --all_path 

python train_search.py  --dataset ACM --num-hops 5 --amp   --all_path
```



## Training stage

After search stage, you can add you search result in `arch.py`, and change the `args.arch`.

You can also **directly run the following command** to get the results in our paper with our searched meta-paths.

```bash
seeds='1 2 3 4 5 6 7 8 9 10'

python main_path.py  --dataset DBLP  --num-hops 6  --residual --amp --seeds $seeds --arch dblp 

python main_path.py  --dataset IMDB  --num-hops 5  --n-layers-2 4 --amp --seeds $seeds  --arch imdb   

python main_path.py  --dataset ACM  --num-hops 5   --amp --seeds $seeds --arch acm  --dropout 0.55

```