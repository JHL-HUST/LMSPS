import os
import gc
import re
import time
import uuid
import argparse
import datetime
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_sparse import remove_diag, set_diag

from model import *
from utils import *
from sparse_tools import SparseAdjList
import numpy as np

# from model import *
from utils import *
def parse_args(args=None):
    parser = argparse.ArgumentParser(description='SeHGNN')
    ## For environment costruction
    parser.add_argument("--seeds", nargs='+', type=int, default=[1],
                        help="the seed used in the training")
    parser.add_argument("--dataset", type=str, default="DBLP")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action='store_true', default=False)
    parser.add_argument("--root", type=str, default="../data/")
    parser.add_argument("--stage", type=int, default=100, help="The epoch setting for each stage.")  # default 200
    parser.add_argument("--embed-size", type=int, default=256,
                        help="inital embedding size of nodes with no attributes")
    parser.add_argument("--num-hops", type=int, default=2,
                        help="number of hops for propagation of raw labels")
    parser.add_argument("--label-feats", action='store_true', default=False,
                        help="whether to use the label propagated features")
    parser.add_argument("--num-label-hops", type=int, default=2,
                        help="number of hops for propagation of raw features")
    ## For network structure
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout on activation")
    parser.add_argument("--n-layers-1", type=int, default=2,
                        help="number of layers of feature projection")
    parser.add_argument("--n-layers-2", type=int, default=3,
                        help="number of layers of the downstream task")
    parser.add_argument("--input-drop", type=float, default=0.1,
                        help="input dropout of input features")
    parser.add_argument("--att-drop", type=float, default=0.,
                        help="attention dropout of model")
    parser.add_argument("--label-drop", type=float, default=0.,
                        help="label feature dropout of model")
    parser.add_argument("--residual", action='store_true', default=False,
                        help="whether to add residual branch the raw input features")
    parser.add_argument("--act", type=str, default='relu',
                        help="the activation function of the model")
    parser.add_argument("--bns", action='store_true', default=False,
                        help="whether to process the input features")
    parser.add_argument("--label-bns", action='store_true', default=False,
                        help="whether to process the input label features")
    ## for training
    parser.add_argument("--amp", action='store_true', default=False,
                        help="whether to amp to accelerate training with float16(half) calculation")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--patience", type=int, default=100,
                        help="early stop of times of the experiment")
    parser.add_argument("--drop-metapath", type=float, default=0,
                        help="whether to process the input features")
    ## for ablation
    parser.add_argument("-na", "--neighbor-attention", action='store_true', default=False)
    parser.add_argument("--nh", type=int, default=1)
    parser.add_argument("--two-layer", action='store_true', default=False)
    parser.add_argument("--remove-transformer", action='store_true', default=False)
    parser.add_argument("--independent-attn", action='store_true', default=False)
    parser.add_argument("--dy", action='store_true', default=False)
    parser.add_argument("--no_path", nargs='+', type=str, default=[])
    parser.add_argument("--no_label", nargs='+', type=str, default=[])
    parser.add_argument("--edge_mask_ratio", type=float, default=0.0)
    parser.add_argument("--label_hop", type=int, default=18)
    parser.add_argument("--ACM_keep_F", type=int, default=False)


    return parser.parse_args(args)

def sp_normalize(mx):
    '''Row-normalize sparse matrix'''
    # 矩阵行求和
    rowsum = np.array(mx.sum(1))
    # 求和的-1次方
    r_inv = np.power(rowsum.astype(float), -1).flatten()
    # 如果是inf，转换成0
    r_inv[np.isinf(r_inv)] = 0
    # 构建对角形矩阵
    r_mat_inv = sp.diags(r_inv)
    # 构造D-I*A, 非对称方式, 简化方式
    mx = r_mat_inv.dot(mx)
    return mx

def average_diameter(g,target_node = 'P',max_hop = 1):
    import  scipy.sparse as sp
    from collections import defaultdict,deque
    from joblib import Parallel, delayed
    my_adjs = {}
    in_links = defaultdict(list)
    for etype in g.etypes:
        sparse_coo = g.adj(etype=etype).coalesce()
        sparse_csr = sp.csr_matrix((sparse_coo.values(),sparse_coo.indices()),shape = sparse_coo.shape)
        sparse_csr = sp_normalize(sparse_csr)
        my_adjs[etype] = sparse_csr
        in_links[etype[0]].append(etype)

        
    total_node_num = g.nodes[target_node].data[target_node].shape[0]

    out_dict = defaultdict(list)
    sample_list = np.random.choice(total_node_num, 20,replace=True).tolist() 
    for index in tqdm(sample_list):
        out_dict[target_node].append(1/total_node_num)
        the_queue = deque()
        init_csr = sp.csr_matrix(([1],([0],[index])),shape=(1,total_node_num))
        the_queue.append((target_node,init_csr))
        while  len(the_queue) >0 :
            item = the_queue.popleft()
            if len(item[0]) <= max_hop:
                for links in in_links[item[0][-1]]:
                    new_str = item[0]+links[-1]
                    new_csr = item[1].dot(my_adjs[links])
                    the_queue.append((new_str,new_csr))

                    out_dict[new_str].append(float(new_csr.data.shape[0])/float(new_csr.shape[1]))
   
    new_out_dict =  defaultdict(list)
    for key in out_dict.keys():
        out_dict[key] = np.mean(out_dict[key])
        new_out_dict[len(key)].append(out_dict[key])

    for key in new_out_dict.keys():
        new_out_dict[key] = np.mean(new_out_dict[key])

    

    return out_dict,new_out_dict


if __name__ == '__main__':
    args = parse_args()
    g, adjs, init_labels, num_classes, dl, train_nid, val_nid, test_nid, test_nid_full \
        = load_dataset(args)
    
    if args.dataset == 'DBLP':
        tgt_type = 'A'
        node_types = ['A', 'P', 'T', 'V']
        extra_metapath = []
    elif args.dataset == 'ACM':
        tgt_type = 'P'
        node_types = ['P', 'A', 'C']
        extra_metapath = []
    elif args.dataset == 'IMDB':
        tgt_type = 'M'
        node_types = ['M', 'A', 'D', 'K']
        extra_metapath = []
    # tosave = propagate_self_value_split_more(g,'P',args.label_hop,args.split_total,args.split_index,10)
    dict0,dict1 = average_diameter(g,tgt_type,args.label_hop)
    print(dict0)
    print(dict1)
    torch.save((dict0,dict1),'out/dict_'+str(args.label_hop)+'.pt')
    
