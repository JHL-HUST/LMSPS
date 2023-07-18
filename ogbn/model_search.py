import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class Transformer(nn.Module):
    def __init__(self, n_channels, att_drop=0., act='none', num_heads=1):
        super(Transformer, self).__init__()
        self.n_channels = n_channels
        self.num_heads = num_heads
        assert self.n_channels % (self.num_heads * 4) == 0

        self.query = nn.Linear(self.n_channels, self.n_channels//4)
        self.key   = nn.Linear(self.n_channels, self.n_channels//4)
        self.value = nn.Linear(self.n_channels, self.n_channels)

        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.att_drop = nn.Dropout(att_drop)
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)
        elif act == 'none':
            self.act = lambda x: x
        else:
            assert 0, f'Unrecognized activation function {act} for class Transformer'

    def reset_parameters(self):

        def xavier_uniform_(tensor, gain=1.):
            fan_in, fan_out = tensor.size()[-2:]
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            return torch.nn.init._no_grad_uniform_(tensor, -a, a)

        gain = nn.init.calculate_gain("relu")
        xavier_uniform_(self.query.weight, gain=gain)
        xavier_uniform_(self.key.weight, gain=gain)
        xavier_uniform_(self.value.weight, gain=gain)
        nn.init.zeros_(self.query.bias)
        nn.init.zeros_(self.key.bias)
        nn.init.zeros_(self.value.bias)

    def forward(self, x, mask=None):
        B, M, C = x.size() # batchsize, num_metapaths, channels
        H = self.num_heads
        if mask is not None:
            assert mask.size() == torch.Size((B, M))

        f = self.query(x).view(B, M, H, -1).permute(0,2,1,3) # [B, H, M, -1]
        g = self.key(x).view(B, M, H, -1).permute(0,2,3,1)   # [B, H, -1, M]
        h = self.value(x).view(B, M, H, -1).permute(0,2,1,3) # [B, H, M, -1]

        beta = F.softmax(self.act(f @ g / math.sqrt(f.size(-1))), dim=-1) # [B, H, M, M(normalized)]
        beta = self.att_drop(beta)
        if mask is not None:
            beta = beta * mask.view(B, 1, 1, M)
            beta = beta / (beta.sum(-1, keepdim=True) + 1e-12)

        o = self.gamma * (beta @ h) # [B, H, M, -1]
        return o.permute(0,2,1,3).reshape((B, M, C)) + x


class Identity(nn.Module):
    "Self attention layer for `n_channels`."
    def __init__(self, n_channels, att_drop=0., act='none', num_heads=1):
        super(Identity, self).__init__()
        self.linear = nn.Identity()#nn.Linear(n_channels, n_channels)

    def forward(self, x, mask=None):

        return self.linear(x)



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

class L2Norm(nn.Module):

    def __init__(self, dim):
        super(L2Norm, self).__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)


class SeHGNN_mag(nn.Module):
    def __init__(self, dataset, data_size, nfeat, hidden, nclass,
                 num_feats, num_label_feats, tgt_key,
                 dropout, input_drop, att_drop, label_drop,
                 n_layers_1, n_layers_2, n_layers_3,
                 act, device, residual=False, bns=False, label_bns=False,
                 label_residual=True, identity=False, num_sampled=0, num_label=0):
        super(SeHGNN_mag, self).__init__()
        self.num_sampled = num_sampled
        self.label_sampled = num_label if num_label_feats else 0
        # self.num_channels = num_sampled
        self.dataset = dataset
        self.residual = residual
        self.tgt_key = tgt_key
        self.label_residual = label_residual
        self.nfeat = nfeat
        self.num_feats = num_feats
        self.num_label_feats = num_label_feats
        self.num_paths = num_feats + num_label_feats


        ####keep the meta paths with features of nfeat as they are####
        if any([v != nfeat for k, v in data_size.items()]):
            self.embedings = nn.ParameterDict({})
            for k, v in data_size.items():
                if v != nfeat:
                    self.embedings[k] = nn.Parameter(
                        torch.Tensor(v, nfeat).uniform_(-0.5, 0.5))
        else:
            self.embedings = None



        self.feat_project_layers = nn.Sequential(
            Conv1d1x1(nfeat, hidden, num_sampled-self.label_sampled, bias=True, cformat='channel-first'),
            nn.LayerNorm([num_sampled-self.label_sampled, hidden]),
            nn.PReLU(),
            nn.Dropout(dropout),
            # Conv1d1x1(hidden, hidden, num_sampled-self.label_sampled, bias=True, cformat='channel-first'),
            # nn.LayerNorm([num_sampled-self.label_sampled, hidden]),
            # nn.PReLU(),
            # nn.Dropout(dropout),
        )
        if num_label_feats > 0:
            self.label_feat_project_layers = nn.Sequential(
                Conv1d1x1(nclass, hidden, self.label_sampled, bias=True, cformat='channel-first'),
                nn.LayerNorm([self.label_sampled, hidden]),
                nn.PReLU(),
                nn.Dropout(dropout),
                # Conv1d1x1(hidden, hidden, self.label_sampled, bias=True, cformat='channel-first'),
                # nn.LayerNorm([self.label_sampled, hidden]),
                # nn.PReLU(),
                # nn.Dropout(dropout),
            )
        else:
            self.label_feat_project_layers = None

        if identity:
            self.semantic_aggr_layers = Identity(hidden, att_drop, act)
        else:
            self.semantic_aggr_layers = Transformer(hidden, att_drop, act)

        if self.dataset != 'products':
            self.concat_project_layer = nn.Identity() #nn.Linear(hidden, hidden) # nn.Linear((num_sampled) * hidden, hidden)  #num_feats + num_label_feats

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



        print ("number of paths", self.num_paths)
        # self.alpha = 1e-3 * torch.randn(self.num_paths).to(device)
        self.alpha = torch.ones(self.num_paths).to(device)

        # self.alpha.cuda()
        self.alpha.requires_grad_(True)

        self.reset_parameters()


    def alphas(self):
        alphas= [self.alpha]
        return alphas

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
            # nn.init.xavier_uniform_(self.concat_project_layer.weight, gain=gain)
            # nn.init.zeros_(self.concat_project_layer.bias)
            ...

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

    def forward(self, feats_dict, epoch_sampled, label_feats_dict, label_emb):
        # meta_path = [k for k, v in feats_dict.items()]
        # label_meta_path = [k for k, v in label_feats_dict.items()]
        # meta_path_sampled = [meta_path[i] for i in range(self.num_feats) if i in epoch_sampled]
        # label_meta_path_sampled = [label_meta_path[i-self.num_feats] for i in range(self.num_feats, self.num_paths) if i in epoch_sampled]
        # id_sampled = [i for i in range(self.num_paths) if i in epoch_sampled]

        #################
        all_meta_path = [k for k, v in feats_dict.items()] + [k for k, v in label_feats_dict.items()]
        meta_path_sampled = [all_meta_path[i] for i in range(self.num_feats) if i in epoch_sampled]
        label_meta_path_sampled = [all_meta_path[i] for i in range(self.num_feats, self.num_paths) if i in epoch_sampled]
        # label_meta_path = [k for k, v in label_feats_dict.items()]


        if self.embedings is not None:
            #only selected meta path and the tgt meta path are involved in the calculation
            for k, v in feats_dict.items():
                if k in self.embedings and k in meta_path_sampled or k==self.tgt_key:
                    feats_dict[k] = v @ self.embedings[k]



        tgt_feat = self.input_drop(feats_dict[self.tgt_key])
        B = num_node = tgt_feat.size(0)

        # feats_dict_selected = [v for k,v in feats_dict.items() if k in meta_path_sampled]
        feats_dict_selected = [feats_dict[k] for k in meta_path_sampled]


        x = self.input_drop(torch.stack(feats_dict_selected, dim=1))


        x = self.feat_project_layers(x)


        if self.label_feat_project_layers is not None:
            # label_feats_dict_selected = [v for k, v in label_feats_dict.items() if k in label_meta_path_sampled]
            label_feats_dict_selected = [label_feats_dict[k] for k in label_meta_path_sampled]

            label_feats = self.input_drop(torch.stack(label_feats_dict_selected, dim=1))
            label_feats = self.label_feat_project_layers(label_feats)
            x = torch.cat((x, label_feats), dim=1)


        #################
        # import code
        # code.interact(local=locals())


        # ws = F.softmax(self.alpha, dim=-1)
        # x = [ws[epoch_sampled[i]]*x[:,i,:] for i in range(len(epoch_sampled))]
        # x = torch.stack(x, dim=1)
        #################


        x = self.semantic_aggr_layers(x)
        if self.dataset == 'products':
            x = x[:,:,0].contiguous()
        else:

            ws = [self.alpha[sample_idx] for sample_idx in epoch_sampled]
            ws = F.softmax(torch.stack(ws), dim=-1)
            x = torch.einsum("bcd,c->bd", x, ws)


            x = self.concat_project_layer(x)#self.concat_project_layer(x.reshape(B, -1))

        if self.residual:
            x = x + self.res_fc(tgt_feat)
        x = self.dropout(self.prelu(x))
        x = self.lr_output(x)
        if self.label_residual:
            x = x + self.label_fc(self.label_drop(label_emb))
        return x

    def epoch_sample(self):
        meta_sampled = random.sample(range(self.num_feats), self.num_sampled-self.label_sampled)
        if self.label_sampled:
            label_sampled = random.sample(range(self.num_feats, self.num_paths), self.label_sampled)
        else:
            label_sampled = []
        sampled = sorted(meta_sampled) + sorted(label_sampled)    #random.sample(range(self.num_paths), self.num_sampled)
        print (sampled)
        return sampled


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
        _, idxl = torch.sort(seq_softmax[:self.num_feats], descending=True)  # descending为False，升序，为True，降序
        _, label_idxl = torch.sort(seq_softmax[self.num_feats:], descending=True)

        idx = list(idxl[:self.num_sampled-self.label_sampled])+list(label_idxl[:self.label_sampled]+self.num_feats)
        idx = sorted(idx)
        # import code
        # code.interact(local=locals())
        if topn:
            id_paths = list(idxl[:topn-self.label_sampled])+list(label_idxl[:self.label_sampled]+self.num_feats)   #idxl[:topn]
        else:
            id_paths = [k for k in range(length) if seq_softmax[k].item() >= threshold]

            #id_paths = [k for k in range(length) if seq_softmax[k].item() >= threshold]
        path = [keys[i] for i in range(len(keys)) if i in id_paths]
        label_path = [label_keys[i] for i in range(len(label_keys)) if i+len(keys) in id_paths]

        if all_path:
            # all_path = []
            # for i in idxl:
            #     all_path.append(keys[i])
            all_path = [keys[i] for i in idxl]
            all_label_path = [label_keys[i] for i in label_idxl]
            return [path, label_path], [all_path, all_label_path], idx
        #print ('seq_softmax', seq_softmax)
        # print (idx)
        # if len(idx)!=4:
        # import code
        # code.interact(local=locals())
        return [path, label_path], idx