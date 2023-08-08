import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor


class Conv1d1x1(nn.Module):
    def __init__(self, cin, cout, groups, bias=True, cformat='channel-first'):
        super(Conv1d1x1, self).__init__()
        self.cin = cin
        self.cout = cout
        self.groups = groups
        self.cformat = cformat
        if not bias:
            self.bias = None
        if self.groups == 1: # different keypoints share same kernel
            self.W = nn.Parameter(torch.randn(self.cin, self.cout))
            if bias:
                self.bias = nn.Parameter(torch.zeros(1, self.cout))
        else:
            self.W = nn.Parameter(torch.randn(self.groups, self.cin, self.cout))
            if bias:
                self.bias = nn.Parameter(torch.zeros(self.groups, self.cout))

    def reset_parameters(self):

        def xavier_uniform_(tensor, gain=1.):
            fan_in, fan_out = tensor.size()[-2:]
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            return torch.nn.init._no_grad_uniform_(tensor, -a, a)

        gain = nn.init.calculate_gain("relu")
        xavier_uniform_(self.W, gain=gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.groups == 1:
            if self.cformat == 'channel-first':
                return torch.einsum('bcm,mn->bcn', x, self.W) + self.bias
            elif self.cformat == 'channel-last':
                return torch.einsum('bmc,mn->bnc', x, self.W) + self.bias.T
            else:
                assert False
        else:
            if self.cformat == 'channel-first':

                # print(x.shape, self.W.shape)
                return torch.einsum('bcm,cmn->bcn', x, self.W) + self.bias
            elif self.cformat == 'channel-last':
                return torch.einsum('bmc,cmn->bnc', x, self.W) + self.bias.T
            else:
                assert False


class LDMLP(nn.Module):
    def __init__(self, nfeat, hidden, nclass, feat_keys, label_feat_keys, tgt_type,
                 dropout, input_drop, att_dropout, label_drop, n_layers_2, residual=False, 
                 bns=False, data_size=None, path=[], label_path=[], eps=0, device=None):
        super(LDMLP, self).__init__()

        self.feat_keys = sorted(feat_keys)
        self.label_feat_keys = sorted(label_feat_keys)
        self.num_channels = num_channels = len(path) + len(label_path)
        self.tgt_type = tgt_type
        self.residual = residual

        self.data_size = data_size 
        self.path = path
        self.label_path = label_path
        self.embeding = nn.ParameterDict({})

        for k, v in data_size.items():
            if k in path or k==tgt_type:
                self.embeding[str(k)] = nn.Parameter(
                torch.Tensor(v, nfeat).uniform_(-0.5, 0.5))
                    
        if len(self.label_feat_keys):
            self.labels_embeding = nn.ParameterDict({})
            for k in self.label_feat_keys:
                if k in label_path:
                    self.labels_embeding[k] = nn.Parameter(
                        torch.Tensor(nclass, nfeat).uniform_(-0.5, 0.5))
        else:
            self.labels_embeding = {}
        

        self.layers = nn.Sequential(
            Conv1d1x1(nfeat, hidden, num_channels, bias=True, cformat='channel-first'),
            nn.LayerNorm([num_channels, hidden]), # nfeat, hidden, num_channels : 512, 512, 9
            nn.PReLU(),
            nn.Dropout(dropout),
            Conv1d1x1(hidden, hidden, num_channels, bias=True, cformat='channel-first'),
            nn.LayerNorm([num_channels, hidden]),
            nn.PReLU(),
            nn.Dropout(dropout),
        )

        self.layer_final = nn.Linear(num_channels * hidden, hidden)

        if self.residual:
            self.res_fc = nn.Linear(nfeat, hidden, bias=False)

        def add_nonlinear_layers(nfeats, dropout, bns=False):
            return [
                nn.BatchNorm1d(hidden, affine=bns, track_running_stats=bns),
                nn.PReLU(),
                nn.Dropout(dropout)
            ]

        lr_output_layers = [
            [nn.Linear(hidden, hidden, bias=not bns)] + add_nonlinear_layers(hidden, dropout, bns)
            for _ in range(n_layers_2-1)]
        self.lr_output = nn.Sequential(*(
            [ele for li in lr_output_layers for ele in li] + [
            nn.Linear(hidden, nclass, bias=False),
            nn.BatchNorm1d(nclass, affine=bns, track_running_stats=bns)]))

        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)  # input_drop=0.1
        self.att_drop = nn.Dropout(att_dropout)
        self.label_drop = nn.Dropout(label_drop)
        #self.dyalpha = Dy(nfeat)
        self.reset_parameters()
        self.epsilon = torch.FloatTensor([eps]).to(device)  #1e-12


    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, Conv1d1x1):
                layer.reset_parameters()

        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.layer_final.weight, gain=gain)
        nn.init.zeros_(self.layer_final.bias)
        if self.residual:
            nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
        for layer in self.lr_output:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, batch, feature_dict, label_dict={}, mask=None):
        if isinstance(feature_dict[self.tgt_type], torch.Tensor):
            mapped_feats = {k: self.input_drop(x @ self.embeding[k]) for k, x in feature_dict.items()}  # @矩阵-向量乘法
        elif isinstance(feature_dict[self.tgt_type], SparseTensor):
            mapped_feats = {k: self.input_drop(x @ self.embeding[k[-1]]) for k, x in feature_dict.items()}
        else:
            assert 0

        mapped_label_feats = {k: self.input_drop(x @ self.labels_embeding[k]) for k, x in label_dict.items()}

        if self.tgt_type in self.path:
            features = [mapped_feats[k] for k in self.feat_keys] + [mapped_label_feats[k] for k in self.label_feat_keys]
        else:
            features = [mapped_feats[k] for k in self.feat_keys if k!=self.tgt_type] + [mapped_label_feats[k] for k in self.label_feat_keys]



        B = num_node = features[0].shape[0] #mapped_feats[self.tgt_type].shape[0] # B: 974
        C = self.num_channels                               # C: 9
        D = features[0].shape[1]    #mapped_feats[self.tgt_type].shape[1]            # D: 512

        features = torch.stack(features, dim=1) # [B, C, D]

        features = self.layers(features).transpose(1,2)

        out = self.layer_final(features.reshape(B, -1))

        if self.residual:
            out = out + self.res_fc(mapped_feats[self.tgt_type])

        # This is an equivalent replacement for tf.l2_normalize
        if self.epsilon:

            out = out / (torch.max(torch.norm(out, dim=1, keepdim=True), self.epsilon))

        out = self.dropout(self.prelu(out))
        out = self.lr_output(out)

        return out

class LDMLP_Se(nn.Module):
    def __init__(self, hidden, nclass, feat_keys, label_feat_keys, tgt_key, dropout, 
                 input_drop, device, residual=False, bns=False, data_size=None, num_sampled=1):
        
        super(LDMLP_Se, self).__init__()

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

        x = torch.einsum('bcd,c->bd', x, ws)

        if self.residual:
            k = self.tgt_key

            tgt_feat = feats_dict[k]
            x = x + self.res_fc(tgt_feat)

        x = self.dropout(self.prelu(x))
        x = self.lr_output(x)
        
        return x

    def sample(self, keys, label_keys, lam, topn, all_path=False):
        length = len(self.alpha)
        seq_softmax = None if self.alpha is None else F.softmax(self.alpha, dim=-1)
        max = torch.max(seq_softmax, dim=0).values
        min = torch.min(seq_softmax, dim=0).values
        threshold = lam * max + (1 - lam) * min

        _, idxl = torch.sort(seq_softmax, descending=True)

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

        return [path, label_path], idx
