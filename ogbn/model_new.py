import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


# class LHMLP_Se(nn.Module):
#     def __init__(self, dataset, data_size, hidden, nclass,
#                  num_feats, num_label_feats, label_feat_keys, tgt_key,
#                  dropout, input_drop, label_drop, device, residual=False, 
#                  label_residual=True, num_sampled=0, num_label=0):
        
#         super(LHMLP_Se, self).__init__()

#         self.num_sampled = num_sampled
#         self.label_sampled = num_label if num_label_feats else 0

#         self.dataset = dataset
#         self.residual = residual
#         self.tgt_key = tgt_key
#         self.label_residual = label_residual

#         self.num_feats = num_feats
#         self.num_label_feats = num_label_feats
#         self.num_paths = num_feats + num_label_feats

#         print("number of paths", num_feats, num_label_feats)

#         self.embeding = nn.ParameterDict({})
#         for k, v in data_size.items():
#             self.embeding[str(k)] = nn.Parameter(
#                 torch.Tensor(v, hidden).uniform_(-0.5, 0.5))

#         if len(label_feat_keys):
#             self.labels_embeding = nn.ParameterDict({})
#             for k in label_feat_keys:
#                 self.labels_embeding[k] = nn.Parameter(
#                     torch.Tensor(nclass, hidden).uniform_(-0.5, 0.5))

#         self.lr_output = nn.Sequential(
#             nn.Linear(hidden, nclass, bias=False),
#             nn.BatchNorm1d(nclass)
#         )

#         self.prelu = nn.PReLU()
#         self.dropout = nn.Dropout(dropout)
#         self.input_drop = nn.Dropout(input_drop)

#         self.alpha = torch.ones(self.num_paths).to(device)
#         self.alpha.requires_grad_(True)

#         if self.residual:
#             self.res_fc = nn.Linear(hidden, hidden)

#         if self.label_residual:
#             self.label_res_fc = nn.Linear(nclass, nclass)
#             self.label_drop = nn.Dropout(label_drop)

#         self.init_params()

#     def init_params(self):

#         gain = nn.init.calculate_gain("relu")

#         for layer in self.lr_output:
#             if isinstance(layer, nn.Linear):
#                 nn.init.xavier_uniform_(layer.weight, gain=gain)
#                 if layer.bias is not None:
#                     nn.init.zeros_(layer.bias)


#     def alphas(self):
#         alphas= [self.alpha]
#         return alphas


#     def epoch_sample(self):
#         meta_sampled = random.sample(range(self.num_feats), self.num_sampled-self.label_sampled)
#         if self.label_sampled:
#             label_sampled = random.sample(range(self.num_feats, self.num_paths), self.label_sampled)
#         else:
#             label_sampled = []
#         sampled = sorted(meta_sampled + label_sampled)
#         print(f"sampled: {sampled}")
#         return sampled
    

#     def forward(self, epoch_sampled, feats_dict, label_feats_dict, label_emb):

#         all_meta_path = list(feats_dict.keys()) + list(label_feats_dict.keys())

#         meta_path_sampled = [all_meta_path[i] for i in range(self.num_feats) if i in epoch_sampled]
#         label_meta_path_sampled = [all_meta_path[i] for i in range(self.num_feats, self.num_paths) if i in epoch_sampled]

#         for k, v in feats_dict.items():
#             if k in self.embeding and k in meta_path_sampled:
#                 feats_dict[k] = self.input_drop(v @ self.embeding[k])
        
#         for k, v in label_feats_dict.items():
#             if k in self.labels_embeding and k in label_meta_path_sampled:
#                 label_feats_dict[k] = self.input_drop(v @ self.labels_embeding[k])

            
#         x = [feats_dict[k] for k in meta_path_sampled] + [label_feats_dict[k] for k in label_meta_path_sampled]
#         x = torch.stack(x, dim=1) # [B, C, D]

#         ws = [self.alpha[idx] for idx in epoch_sampled]
#         ws = F.softmax(torch.stack(ws), dim=-1)

#         # import code
#         # code.interact(local=locals())

#         x = torch.einsum('bcd,c->bd', x, ws)

#         if self.residual:
#             k = self.tgt_key
#             if k not in meta_path_sampled:
#                 tgt_feat = self.input_drop(feats_dict[k] @ self.embeding[k])
#             else:
#                 tgt_feat = feats_dict[k]
#             x = x + self.res_fc(tgt_feat)

#         x = self.dropout(self.prelu(x))
#         x = self.lr_output(x)
        
#         if self.label_residual:
#             x = x + self.label_res_fc(self.label_drop(label_emb))

#         return x

#     def sample(self, keys, label_keys, lam, topn, all_path=False):
#         '''
#         to sample one candidate edge type per link
#         '''
#         length = len(self.alpha)
#         seq_softmax = None if self.alpha is None else F.softmax(self.alpha, dim=-1)
#         max = torch.max(seq_softmax, dim=0).values
#         min = torch.min(seq_softmax, dim=0).values
#         threshold = lam * max + (1 - lam) * min


#         ##keep the number of selected meta path and selected label meta path to be fixed
#         _, idxl = torch.sort(seq_softmax[:self.num_feats], descending=True)  # descending为False，升序，为True，降序
#         _, label_idxl = torch.sort(seq_softmax[self.num_feats:], descending=True)

#         idx = list(idxl[:self.num_sampled-self.label_sampled])+list(label_idxl[:self.label_sampled]+self.num_feats)
#         idx = sorted(idx)
#         # import code
#         # code.interact(local=locals())
#         if topn:
#             id_paths = list(idxl[:topn-self.label_sampled])+list(label_idxl[:self.label_sampled]+self.num_feats)   #idxl[:topn]
#         else:
#             id_paths = [k for k in range(length) if seq_softmax[k].item() >= threshold]

#             #id_paths = [k for k in range(length) if seq_softmax[k].item() >= threshold]
#         path = [keys[i] for i in range(len(keys)) if i in id_paths]
#         label_path = [label_keys[i] for i in range(len(label_keys)) if i+len(keys) in id_paths]

#         if all_path:
#             # all_path = []
#             # for i in idxl:
#             #     all_path.append(keys[i])
#             all_path = [keys[i] for i in idxl]
#             all_label_path = [label_keys[i] for i in label_idxl]
#             return [path, label_path], [all_path, all_label_path], idx
#         #print ('seq_softmax', seq_softmax)
#         # print (idx)
#         # if len(idx)!=4:
#         # import code
#         # code.interact(local=locals())
#         return [path, label_path], idx



class LHMLP_Se(nn.Module):
    def __init__(self, dataset, data_size, hidden, nclass,
                 num_feats, num_label_feats, label_feat_keys, tgt_key,
                 dropout, input_drop, label_drop, device, residual=False, 
                 label_residual=True, num_sampled=0, num_label=0):
        
        super(LHMLP_Se, self).__init__()

        self.num_sampled = num_sampled
        # self.label_sampled = num_label if num_label_feats else 0

        self.dataset = dataset
        self.residual = residual
        self.tgt_key = tgt_key
        self.label_residual = label_residual

        self.num_feats = num_feats
        self.num_label_feats = num_label_feats
        self.num_paths = num_feats + num_label_feats

        print("number of paths", num_feats, num_label_feats)

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
            nn.BatchNorm1d(nclass)
        )

        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)

        self.alpha = torch.ones(self.num_paths).to(device)
        self.alpha.requires_grad_(True)

        if self.residual:
            self.res_fc = nn.Linear(hidden, hidden)

        if self.label_residual:
            self.label_res_fc = nn.Linear(nclass, nclass)
            self.label_drop = nn.Dropout(label_drop)

        self.init_params()

    def init_params(self):

        gain = nn.init.calculate_gain("relu")

        for layer in self.lr_output:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)


    def alphas(self):
        alphas= [self.alpha]
        return alphas


    def epoch_sample(self):
        # meta_sampled = random.sample(range(self.num_feats), self.num_sampled-self.label_sampled)
        # if self.label_sampled:
        #     label_sampled = random.sample(range(self.num_feats, self.num_paths), self.label_sampled)
        # else:
        #     label_sampled = []
        # sampled = sorted(meta_sampled + label_sampled)
        sampled = random.sample(range(self.num_paths), self.num_sampled)
        sampled = sorted(sampled)
        print(f"sampled: {sampled}")
        return sampled
    

    def forward(self, epoch_sampled, feats_dict, label_feats_dict, label_emb):

        all_meta_path = list(feats_dict.keys()) + list(label_feats_dict.keys())

        meta_path_sampled = [all_meta_path[i] for i in range(self.num_feats) if i in epoch_sampled]
        label_meta_path_sampled = [all_meta_path[i] for i in range(self.num_feats, self.num_paths) if i in epoch_sampled]

        for k, v in feats_dict.items():
            if k in self.embeding and k in meta_path_sampled:
                feats_dict[k] = self.input_drop(v @ self.embeding[k])
        
        for k, v in label_feats_dict.items():
            if k in self.labels_embeding and k in label_meta_path_sampled:
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
            if k not in meta_path_sampled:
                tgt_feat = self.input_drop(feats_dict[k] @ self.embeding[k])
            else:
                tgt_feat = feats_dict[k]
            x = x + self.res_fc(tgt_feat)

        x = self.dropout(self.prelu(x))
        x = self.lr_output(x)
        
        if self.label_residual:
            x = x + self.label_res_fc(self.label_drop(label_emb))

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


        ##keep the number of selected meta path and selected label meta path to be fixed
        # _, idxl = torch.sort(seq_softmax[:self.num_feats], descending=True)  # descending为False，升序，为True，降序
        # _, label_idxl = torch.sort(seq_softmax[self.num_feats:], descending=True)
        _, idxl = torch.sort(seq_softmax, descending=True)  # descending为alse，升序，为True，降序


        # idx = list(idxl[:self.num_sampled-self.label_sampled])+list(label_idxl[:self.label_sampled]+self.num_feats)
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

        # idx = sorted(idx)
        # # import code
        # # code.interact(local=locals())
        # if topn:
        #     id_paths = list(idxl[:topn-self.label_sampled])+list(label_idxl[:self.label_sampled]+self.num_feats)   #idxl[:topn]
        # else:
        #     id_paths = [k for k in range(length) if seq_softmax[k].item() >= threshold]

        #     #id_paths = [k for k in range(length) if seq_softmax[k].item() >= threshold]
        # path = [keys[i] for i in range(len(keys)) if i in id_paths]
        # label_path = [label_keys[i] for i in range(len(label_keys)) if i+len(keys) in id_paths]

        # if all_path:
        #     # all_path = []
        #     # for i in idxl:
        #     #     all_path.append(keys[i])
        #     all_path = [keys[i] for i in idxl]
        #     all_label_path = [label_keys[i] for i in label_idxl]
        #     return [path, label_path], [all_path, all_label_path], idx
        # #print ('seq_softmax', seq_softmax)
        # # print (idx)
        # # if len(idx)!=4:
        # # import code
        # # code.interact(local=locals())
        # return [path, label_path], idx


