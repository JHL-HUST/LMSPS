import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from dy import Dy, DyReLUA, DyReLUB


class Transformer(nn.Module):
    "Self attention layer for `n_channels`."
    def __init__(self, n_channels, num_heads=1, att_drop=0., act='none'):
        super(Transformer, self).__init__()
        self.n_channels = n_channels
        self.num_heads = num_heads
        assert self.n_channels % (self.num_heads * 4) == 0

        self.query = nn.Linear(self.n_channels, self.n_channels//4)   # original 4
        self.key   = nn.Linear(self.n_channels, self.n_channels//4)   # original 4
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
        elif act == 'dy_relu1':
            self.act = DyReLUA(self.num_heads)
        elif act == 'dy_relu2':
            self.act = DyReLUB(self.num_heads)
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
        beta = self.att_drop(beta)  # p=0
        if mask is not None:
            beta = beta * mask.view(B, 1, 1, M)
            beta = beta / (beta.sum(-1, keepdim=True) + 1e-12)

        o = self.gamma * (beta @ h) # [B, H, M, -1]
        return o.permute(0,2,1,3).reshape((B, M, C)) + x


class Identity(nn.Module):
    "Self attention layer for `n_channels`."
    def __init__(self, n_channels, num_heads=1, att_drop=0., act='none', proj_dim=None):
        super(Identity, self).__init__()

    def forward(self, x, mask=None):
        return x


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128, independent_attn=False):
        super(SemanticAttention, self).__init__()
        self.independent_attn = independent_attn

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z): # z: (N, M, D)
        w = self.project(z) # (N, M, 1)
        if not self.independent_attn:
            w = w.mean(0, keepdim=True)  # (1, M, 1)
        beta = torch.softmax(w, dim=1)  # (N, M, 1) or (1, M, 1)

        return (beta * z).sum(1)  # (N, M, D)


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


class SeHGNN(nn.Module):
    def __init__(self, nfeat, hidden, nclass, feat_keys, label_feat_keys, tgt_type,
                 dropout, input_drop, att_dropout, label_drop,
                 n_layers_1, n_layers_2, act,
                 residual=False, identity=False, bns=False, data_size=None, drop_metapath=0., num_heads=1, path=[], label_path=[], eps=0, device=None,
                 remove_transformer=True, independent_attn=False, dataset=''):
        super(SeHGNN, self).__init__()

        self.feat_keys = sorted(feat_keys)
        self.label_feat_keys = sorted(label_feat_keys)
        self.num_channels = num_channels = len(path) + len(label_path)
        self.tgt_type = tgt_type
        self.residual = residual
        self.remove_transformer = remove_transformer

        self.data_size = data_size  #{'A': 334, 'AP': 4231, 'APA': 334, 'APT': 50, 'APV': 20}
        self.path = path
        self.label_path = label_path
        self.embeding = nn.ParameterDict({})

        # import code
        # code.interact(local=locals())

        for k, v in data_size.items():
            if dataset=='Freebase':
                self.embeding[str(k)] = nn.Parameter(
                    torch.Tensor(v, nfeat).uniform_(-0.5, 0.5))
            else:
                if k in path or k==tgt_type:
                    self.embeding[str(k)[-1]] = nn.Parameter(
                    torch.Tensor(v, nfeat).uniform_(-0.5, 0.5))
                    
        if len(self.label_feat_keys):
            self.labels_embeding = nn.ParameterDict({})
            for k in self.label_feat_keys:
                if k in label_path:
                    self.labels_embeding[k] = nn.Parameter(
                        torch.Tensor(nclass, nfeat).uniform_(-0.5, 0.5))
        else:
            self.labels_embeding = {}
        
        # import code
        # code.interact(local=locals())

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


        if self.remove_transformer:
            self.layer_mid = SemanticAttention(hidden, hidden, independent_attn=independent_attn)
            self.layer_final = nn.Linear(hidden, hidden)
        else:
            if identity:
                self.layer_mid = Identity(hidden, num_heads=num_heads)
            else:
                self.layer_mid = Transformer(hidden, num_heads=num_heads)
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
        # import code
        # code.interact(local=locals())


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
            mapped_feats = {k: self.input_drop(x @ self.embeding[k[-1]]) for k, x in feature_dict.items()}  # @矩阵-向量乘法
            # DBLP: {'A': 334, 'AP': 4231, 'APA': 334, 'APT': 50, 'APV': 20}
            # IMDB(25): ['M', 'MA', 'MD', 'MK', 'MAM', 'MDM', 'MKM', 'MAMA', 'MAMD', 'MAMK', 'MDMA', 'MDMD', 'MDMK', 'MKMA', 'MKMD', 'MKMK', 'MAMAM', 'MAMDM', 'MAMKM', 'MDMAM', 'MDMDM', 'MDMKM', 'MKMAM', 'MKMDM', 'MKMKM']
            # ACM(41): ['P', 'PA', 'PC', 'PP', 'PAP', 'PCP', 'PPA', 'PPC', 'PPP', 'PAPA', 'PAPC', 'PAPP', 'PCPA', 'PCPC', 'PCPP', 'PPAP', 'PPCP', 'PPPA', 'PPPC', 'PPPP', 'PAPAP', 'PAPCP', 'PAPPA', 'PAPPC', 'PAPPP', 'PCPAP', 'PCPCP', 'PCPPA', 'PCPPC', ,'PCPPP', 'PPAPA', 'PPAPC', 'PPAPP', 'PPCPA', 'PPCPC', 'PPPCPP', 'PPPAP', 'PPPCP', 'PPPPA', 'PPPPC', 'PPPPP']
        elif isinstance(feature_dict[self.tgt_type], SparseTensor):

            mapped_feats = {k: self.input_drop(x @ self.embeding[k[-1][-1]]) for k, x in feature_dict.items()}
            # ['APA', 'APAPA', 'APTPA', 'APVPA']
            # IMDB(12): ['MDM', 'MAM', 'MKM', 'MDMDM', 'MAMDM', 'MKMDM', 'MDMAM', 'MAMAM', 'MKMAM', 'MDMKM', 'MAMKM', 'MKMKM']
            # ACM(9): ['PP', 'PPP', 'PAP', 'PCP', 'PPPP', 'PAPP', 'PCPP', 'PPAP', 'PPCP']
        else:
            assert 0

        mapped_label_feats = {k: self.input_drop(x @ self.labels_embeding[k]) for k, x in label_dict.items()}


        #dyalpha = self.dyalpha(mapped_feats)
        if self.tgt_type in self.path:
            features = [mapped_feats[k] for k in self.feat_keys] + [mapped_label_feats[k] for k in self.label_feat_keys]
        else:
            features = [mapped_feats[k] for k in self.feat_keys if k!=self.tgt_type] + [mapped_label_feats[k] for k in self.label_feat_keys]
        # [974,512]  self.dyalpha(mapped_label_feats[k])   if k in self.path    if k in self.label_path



        B = num_node = features[0].shape[0] #mapped_feats[self.tgt_type].shape[0] # B: 974
        C = self.num_channels                               # C: 9
        D = features[0].shape[1]    #mapped_feats[self.tgt_type].shape[1]            # D: 512

        features = torch.stack(features, dim=1) # [B, C, D]

        # import code
        # code.interact(local=locals())


        features = self.layers(features)
        if self.remove_transformer:
            features = self.layer_mid(features)
        else:
            features = self.layer_mid(features, mask=None).transpose(1,2)
        out = self.layer_final(features.reshape(B, -1))



        if self.residual:
            out = out + self.res_fc(mapped_feats[self.tgt_type])



        # This is an equivalent replacement for tf.l2_normalize
        if self.epsilon:

            out = out / (torch.max(torch.norm(out, dim=1, keepdim=True), self.epsilon))

        out = self.dropout(self.prelu(out))
        out = self.lr_output(out)




        return out




class SeHGNN_2Linear(nn.Module):
    def __init__(self, nfeat, hidden, nclass, feat_keys, label_feat_keys, tgt_type,
                 dropout, input_drop, att_dropout, label_drop,
                 n_layers_1, n_layers_2, act,
                 residual=False, bns=False, data_size=None, drop_metapath=0., num_heads=1, path=[], label_path=[], eps=0, device=None,
                 remove_transformer=True, independent_attn=False):
        super(SeHGNN_2Linear, self).__init__()

        self.feat_keys = sorted(feat_keys)
        self.label_feat_keys = sorted(label_feat_keys)
        self.num_channels = num_channels = len(path) + len(label_path)
        self.tgt_type = tgt_type
        self.residual = residual
        self.remove_transformer = remove_transformer

        self.data_size = data_size  #{'A': 334, 'AP': 4231, 'APA': 334, 'APT': 50, 'APV': 20}
        self.path = path
        self.label_path = label_path
        self.embeding = nn.ParameterDict({})

        for k, v in data_size.items():
            self.embeding[str(k)] = nn.Parameter(
                torch.Tensor(v, nfeat).uniform_(-0.5, 0.5))

        if len(self.label_feat_keys):
            self.labels_embeding = nn.ParameterDict({})
            for k in self.label_feat_keys:
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


        if self.remove_transformer:
            self.layer_mid = SemanticAttention(hidden, hidden, independent_attn=independent_attn)
            self.layer_final = nn.Linear(hidden, hidden)
        else:
            self.layer_mid = nn.Sequential(Transformer(hidden, num_heads=num_heads),
                                           nn.Dropout(0.1),
                                           Transformer(hidden, num_heads=num_heads),
                                           nn.Dropout(0.1)
                                           )

            # Transformer(hidden, num_heads=num_heads),
            # Transformer(hidden, num_heads=num_heads),# nfeat, hidden, num_channels : 512, 512, 9

            Transformer(hidden, num_heads=num_heads)
            # import code
            # code.interact(local=locals())
            # self.layer_mid1 = Transformer(hidden, num_heads=num_heads)
            # self.layer_mid2 = Transformer(hidden, num_heads=num_heads)

            self.layer_final = nn.Linear(num_channels * hidden, hidden)
            # self.layer_final2 = nn.Linear(hidden, hidden)

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
        self.trans_dropout = nn.Dropout(0.1)
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

        # nn.init.xavier_uniform_(self.layer_final2.weight, gain=gain)
        # nn.init.zeros_(self.layer_final2.bias)

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
            # DBLP: {'A': 334, 'AP': 4231, 'APA': 334, 'APT': 50, 'APV': 20}
            # IMDB(25): ['M', 'MA', 'MD', 'MK', 'MAM', 'MDM', 'MKM', 'MAMA', 'MAMD', 'MAMK', 'MDMA', 'MDMD', 'MDMK', 'MKMA', 'MKMD', 'MKMK', 'MAMAM', 'MAMDM', 'MAMKM', 'MDMAM', 'MDMDM', 'MDMKM', 'MKMAM', 'MKMDM', 'MKMKM']
            # ACM(41): ['P', 'PA', 'PC', 'PP', 'PAP', 'PCP', 'PPA', 'PPC', 'PPP', 'PAPA', 'PAPC', 'PAPP', 'PCPA', 'PCPC', 'PCPP', 'PPAP', 'PPCP', 'PPPA', 'PPPC', 'PPPP', 'PAPAP', 'PAPCP', 'PAPPA', 'PAPPC', 'PAPPP', 'PCPAP', 'PCPCP', 'PCPPA', 'PCPPC', ,'PCPPP', 'PPAPA', 'PPAPC', 'PPAPP', 'PPCPA', 'PPCPC', 'PPPCPP', 'PPPAP', 'PPPCP', 'PPPPA', 'PPPPC', 'PPPPP']
        elif isinstance(feature_dict[self.tgt_type], SparseTensor):
            mapped_feats = {k: self.input_drop(x @ self.embeding[k[-1]]) for k, x in feature_dict.items()}
            # ['APA', 'APAPA', 'APTPA', 'APVPA']
            # IMDB(12): ['MDM', 'MAM', 'MKM', 'MDMDM', 'MAMDM', 'MKMDM', 'MDMAM', 'MAMAM', 'MKMAM', 'MDMKM', 'MAMKM', 'MKMKM']
            # ACM(9): ['PP', 'PPP', 'PAP', 'PCP', 'PPPP', 'PAPP', 'PCPP', 'PPAP', 'PPCP']
        else:
            assert 0

        mapped_label_feats = {k: self.input_drop(x @ self.labels_embeding[k]) for k, x in label_dict.items()}


        #dyalpha = self.dyalpha(mapped_feats)
        if self.tgt_type in self.path:
            features = [mapped_feats[k] for k in self.feat_keys] + [mapped_label_feats[k] for k in self.label_feat_keys]
        else:
            features = [mapped_feats[k] for k in self.feat_keys if k!=self.tgt_type] + [mapped_label_feats[k] for k in self.label_feat_keys]
        # [974,512]  self.dyalpha(mapped_label_feats[k])   if k in self.path    if k in self.label_path



        B = num_node = features[0].shape[0] #mapped_feats[self.tgt_type].shape[0] # B: 974
        C = self.num_channels                               # C: 9
        D = features[0].shape[1]    #mapped_feats[self.tgt_type].shape[1]            # D: 512

        features = torch.stack(features, dim=1) # [B, C, D]

        # import code
        # code.interact(local=locals())


        features = self.layers(features)
        if self.remove_transformer:
            features = self.layer_mid(features)
        else:
            #features = self.layer_mid1(features, mask=None)  #.transpose(1,2)
            features = self.layer_mid(features).transpose(1, 2)  #, mask=None

        features = self.trans_dropout(features)    #self.trans_dropout(self.prelu(features))
        out = self.layer_final(features.reshape(B, -1))
        # out = self.layer_final2(out)

        if self.residual:
            out = out + self.res_fc(mapped_feats[self.tgt_type])

        # This is an equivalent replacement for tf.l2_normalize
        if self.epsilon:
            # import code
            # code.interact(local=locals())
            out = out / (torch.max(torch.norm(out, dim=1, keepdim=True), self.epsilon))

        out = self.dropout(self.prelu(out))
        out = self.lr_output(out)
        return out




