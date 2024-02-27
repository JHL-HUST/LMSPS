import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


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
                return torch.einsum('bcm,cmn->bcn', x, self.W) + self.bias
            elif self.cformat == 'channel-last':
                return torch.einsum('bmc,cmn->bnc', x, self.W) + self.bias.T
            else:
                assert False


class LMSPS(nn.Module):
    def __init__(self, dataset, data_size, nfeat, hidden, nclass,
                 num_feats, num_label_feats, tgt_key,
                 dropout, input_drop, att_drop, label_drop,
                  n_layers_2, n_layers_3,
                 residual=False, bns=False, label_bns=False,
                 label_residual=True, path=[], label_path=[], eps = 0, device = None):
        super(LMSPS, self).__init__()
        self.dataset = dataset
        self.residual = residual
        self.tgt_key = tgt_key
        self.label_residual = label_residual
        self.path = path
        self.label_path = label_path
        self.num_meta = len(path)
        self.num_label_meta = len(label_path)
        if self.num_label_meta > num_label_feats: 
            self.num_label_meta = num_label_feats
        if any([v != nfeat for k, v in data_size.items()]):
            self.embedings = nn.ParameterDict({})
            for k, v in data_size.items():
                if v != nfeat:
                    self.embedings[k] = nn.Parameter(
                        torch.Tensor(v, nfeat).uniform_(-0.5, 0.5))
        else:
            self.embedings = None

        self.feat_project_layers = nn.Sequential(
            Conv1d1x1(nfeat, hidden, self.num_meta, bias=True, cformat='channel-first'),
            nn.LayerNorm([self.num_meta, hidden]),
            nn.PReLU(),
            nn.Dropout(dropout),
            Conv1d1x1(hidden, hidden, self.num_meta, bias=True, cformat='channel-first'),
            nn.LayerNorm([self.num_meta, hidden]),
            nn.PReLU(),
            nn.Dropout(dropout),
        )
        if num_label_feats>0 and label_path:
            self.label_feat_project_layers = nn.Sequential(
                Conv1d1x1(nclass, hidden, self.num_label_meta, bias=True, cformat='channel-first'),
                nn.LayerNorm([self.num_label_meta, hidden]),
                nn.PReLU(),
                nn.Dropout(dropout),
                Conv1d1x1(hidden, hidden, self.num_label_meta, bias=True, cformat='channel-first'),
                nn.LayerNorm([self.num_label_meta, hidden]),
                nn.PReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.label_feat_project_layers = None

        self.concat_project_layer = nn.Linear((self.num_meta + self.num_label_meta) * hidden, hidden)

        if self.residual:
            self.res_fc = nn.Linear(nfeat, hidden, bias=False)

        def add_nonlinear_layers(nfeats, dropout, bns=False):
            if bns:
                return [
                    nn.BatchNorm1d(hidden),
                    nn.PReLU(),
                    nn.Dropout(dropout)
                ]
            else:
                return [
                    nn.PReLU(),
                    nn.Dropout(dropout)
                ]

        lr_output_layers = [
            [nn.Linear(hidden, hidden, bias=not bns)] + add_nonlinear_layers(hidden, dropout, bns)
            for _ in range(n_layers_2-1)]
        self.lr_output = nn.Sequential(*(
            [ele for li in lr_output_layers for ele in li] + [
            nn.Linear(hidden, nclass, bias=False),
            nn.BatchNorm1d(nclass)]))

        if self.label_residual:
            label_fc_layers = [
                [nn.Linear(hidden, hidden, bias=not bns)] + add_nonlinear_layers(hidden, dropout, bns)
                for _ in range(n_layers_3-2)]
            self.label_fc = nn.Sequential(*(
                [nn.Linear(nclass, hidden, bias=not bns)] + add_nonlinear_layers(hidden, dropout, bns) \
                + [ele for li in label_fc_layers for ele in li] + [nn.Linear(hidden, nclass, bias=True)]))
            self.label_drop = nn.Dropout(label_drop)

        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)
        self.epsilon = torch.FloatTensor([eps]).to(device)  #1e-12
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")

        for layer in self.feat_project_layers:
            if isinstance(layer, Conv1d1x1):
                layer.reset_parameters()
        if self.label_feat_project_layers is not None:
            for layer in self.label_feat_project_layers:
                if isinstance(layer, Conv1d1x1):
                    layer.reset_parameters()

        if self.dataset != 'products':
            nn.init.xavier_uniform_(self.concat_project_layer.weight, gain=gain)
            nn.init.zeros_(self.concat_project_layer.bias)

        if self.residual:
            nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)

        for layer in self.lr_output:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        if self.label_residual:
            for layer in self.label_fc:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=gain)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, feats_dict, layer_feats_dict, label_emb):
        if self.embedings is not None:
            for k, v in feats_dict.items():
                if k in self.embedings:
                    feats_dict[k] = v @ self.embedings[k]


        tgt_feat = self.input_drop(feats_dict[self.tgt_key])
        B = num_node = tgt_feat.size(0)

        # meta_path = [v for k,v in feats_dict.items() if k in self.path]
        if self.tgt_key not in self.path:
            feats_dict = {k:v for k, v in feats_dict.items() if k!=self.tgt_key}

        x = self.input_drop(torch.stack(list(feats_dict.values()), dim=1))

        x = self.feat_project_layers(x)

        if self.label_feat_project_layers is not None:
            label_feats = self.input_drop(torch.stack(list(layer_feats_dict.values()), dim=1))

            label_feats = self.label_feat_project_layers(label_feats)
            x = torch.cat((x, label_feats), dim=1)

        x = self.concat_project_layer(x.reshape(B, -1))

        if self.residual:
            x = x + self.res_fc(tgt_feat)
        x = self.dropout(self.prelu(x))
        x = self.lr_output(x)
        if self.label_residual:
            x = x + self.label_fc(self.label_drop(label_emb))
        if self.epsilon:
            x = x / (torch.max(torch.norm(x, dim=1, keepdim=True), self.epsilon))
        return x


class LMSPS_Se(nn.Module):
    def __init__(self, dataset, data_size, hidden, nclass,
                 num_feats, num_label_feats, label_feat_keys, tgt_key,
                 dropout, input_drop, label_drop, device, residual=False, 
                 label_residual=True, num_sampled=0):
        
        super(LMSPS_Se, self).__init__()

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

        length = len(self.alpha)
        seq_softmax = None if self.alpha is None else F.softmax(self.alpha, dim=-1)
        max = torch.max(seq_softmax, dim=0).values
        min = torch.min(seq_softmax, dim=0).values
        threshold = lam * max + (1 - lam) * min


        _, idxl = torch.sort(seq_softmax, descending=True)  # descending为alse，升序，为True，降序

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
