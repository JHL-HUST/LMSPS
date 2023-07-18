import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
import random

class LHMLP_Se(nn.Module):
    def __init__(self, hidden, nclass, feat_keys, label_feat_keys, tgt_key, dropout, 
                 input_drop, device, residual=False, bns=False, data_size=None, num_sampled=1):
        
        super(LHMLP_Se, self).__init__()

        self.feat_keys = feat_keys
        self.label_feat_keys = label_feat_keys
        self.num_feats = len(feat_keys)
        self.all_meta_path = list(self.feat_keys) + list(self.label_feat_keys)
        self.num_sampled = num_sampled
        self.num_channels = self.num_sampled
        self.num_paths = len(self.all_meta_path)

        self.tgt_key = tgt_key
        self.residual = residual

        print("number of paths", len(feat_keys), len(label_feat_keys))

        self.embeding = nn.ParameterDict({})
        for k, v in data_size.items():
            self.embeding[str(k)] = nn.Parameter(
                torch.Tensor(v, hidden).uniform_(-0.5, 0.5))

        if len(label_feat_keys):
            self.labels_embeding = nn.ParameterDict({})
            for k in label_feat_keys:
                self.labels_embeding[k] = nn.Parameter(
                    torch.Tensor(nclass, hidden).uniform_(-0.5, 0.5))

        self.lr_output = nn.Sequential(
            nn.Linear(hidden, nclass, bias=False),
            nn.BatchNorm1d(nclass, affine=bns, track_running_stats=bns)
        )

        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)

        self.alpha = torch.ones(self.num_paths).to(device)
        self.alpha.requires_grad_(True)

        if self.residual:
            self.res_fc = nn.Linear(hidden, hidden)

        self.init_params()

    def init_params(self):

        gain = nn.init.calculate_gain("relu")
        if self.residual:
            nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
        for layer in self.lr_output:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)


    def alphas(self):
        alphas= [self.alpha]
        return alphas


    def epoch_sample(self):
        sampled = random.sample(range(self.num_paths), self.num_sampled)
        sampled = sorted(sampled)
        print(f"sampled: {sampled}")
        return sampled
    

    def forward(self, epoch_sampled, feats_dict, label_feats_dict, meta_path_sampled, label_meta_path_sampled):

        # meta_path_sampled = [self.all_meta_path[i] for i in range(self.num_feats) if i in epoch_sampled]
        # label_meta_path_sampled = [self.all_meta_path[i] for i in range(self.num_feats, self.num_paths) if i in epoch_sampled]

        # for k, v in feats_dict.items():
        #     if k in self.embeding and k in meta_path_sampled:
        #         feats_dict[k] = self.input_drop(v @ self.embeding[k])
        
        # for k, v in label_feats_dict.items():
        #     if k in self.labels_embeding and k in label_meta_path_sampled:
        #         label_feats_dict[k] = self.input_drop(v @ self.labels_embeding[k])

        for k, v in feats_dict.items():
            if k in self.embeding:
                feats_dict[k] = self.input_drop(v @ self.embeding[k])
        
        for k, v in label_feats_dict.items():
            if k in self.labels_embeding:
                label_feats_dict[k] = self.input_drop(v @ self.labels_embeding[k])


            
        x = [feats_dict[k] for k in meta_path_sampled] + [label_feats_dict[k] for k in label_meta_path_sampled]
        x = torch.stack(x, dim=1) # [B, C, D]

        ws = [self.alpha[idx] for idx in epoch_sampled]
        ws = F.softmax(torch.stack(ws), dim=-1)

        # import code
        # code.interact(local=locals())

        x = torch.einsum('bcd,c->bd', x, ws)

        if self.residual:
            k = self.tgt_key
            # if k not in meta_path_sampled:
            #     tgt_feat = self.input_drop(feats_dict[k] @ self.embeding[k])
            # else:
            tgt_feat = feats_dict[k]
            x = x + self.res_fc(tgt_feat)

        x = self.dropout(self.prelu(x))
        x = self.lr_output(x)
        
        return x

    def sample(self, keys, label_keys, lam, topn, all_path=False):
        '''
        to sample one candidate edge type per link
        '''
        length = len(self.alpha)
        seq_softmax = None if self.alpha is None else F.softmax(self.alpha, dim=-1)
        max = torch.max(seq_softmax, dim=0).values
        min = torch.min(seq_softmax, dim=0).values
        threshold = lam * max + (1 - lam) * min


        _, idxl = torch.sort(seq_softmax, descending=True)  # descending为alse，升序，为True，降序
        # import code
        # code.interact(local=locals())
        idx = idxl[:self.num_sampled]

        if all_path:
            path = []
            label_path = []
            for i, index in enumerate(idxl):
                if index < len(keys):
                    path.append((keys[index], i))
                else:
                    label_path.append((label_keys[index - len(keys)], i))
            return [path, label_path], idx

        if topn:
            id_paths = idxl[:topn]
        else:
            id_paths = [k for k in range(length) if seq_softmax[k].item() >= threshold]
        path = [keys[i] for i in range(len(keys)) if i in id_paths]
        label_path = [label_keys[i] for i in range(len(label_keys)) if i+len(keys) in id_paths]
        # import code
        # code.interact(local=locals())
        # print(idxl)
        # print(seq_softmax)
        return [path, label_path], idx




