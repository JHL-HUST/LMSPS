import os
import gc
import random
import dgl
import dgl.function as fn
import numpy as np
import scipy.sparse as sp
import sparse_tools
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch_sparse import SparseTensor
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator



def project_op(keys, label_keys, model, criterion, eval_loader, device, trainval_point, valtest_point, valid_node_nums, labels, repeat):  #
    compare = lambda x, y: x < y   ######################attention########################
    crit_extrema = None
    best_index = 0
    best_id = 0
    for opid in range(repeat):
        print (opid)
<<<<<<< HEAD
        index_sampled = model.epoch_sample(0, keys)
=======
        index_sampled = model.epoch_sample(0)
>>>>>>> c50912ad06c4a232ec40e52336f71357ed08d8b3
        crit = infer_eval(model, criterion, eval_loader, device, index_sampled, trainval_point, valtest_point, valid_node_nums, labels)  #weights=weights
        print("loss {}\n".format(crit))
        if crit_extrema is None or compare(crit, crit_extrema):
            crit_extrema = crit
            best_index = index_sampled
            best_id = opid
    print ("best_id: {}\n".format(best_id))
    path = []
    label_path = []
    for i, index in enumerate(best_index):
        if index < len(keys):
            path.append(keys[index])
        else:
            label_path.append(label_keys[index - len(keys)])

    return [path, label_path]


def infer_eval(model, criterion, eval_loader, device, index_sampled, trainval_point, valtest_point, valid_node_nums, labels):
    with torch.no_grad():
        model.eval()
        raw_preds = []
        meta_path_sampled = [model.all_meta_path[i] for i in range(model.num_feats) if i in index_sampled]
        label_meta_path_sampled = [model.all_meta_path[i] for i in range(model.num_feats,model.num_paths) if i in index_sampled]


        for batch_feats, batch_label_feats, batch_labels_emb in eval_loader:
            batch_feats = {k: v.to(device).float() for k, v in batch_feats.items()}
            batch_label_feats = {k: v.to(device) for k, v in batch_label_feats.items()}
            batch_labels_emb = batch_labels_emb.to(device)
            raw_preds.append(model(index_sampled, batch_feats, batch_label_feats, batch_labels_emb).cpu())
    raw_preds = torch.cat(raw_preds, dim=0)

    loss_val = criterion(raw_preds[:valid_node_nums], labels[trainval_point:valtest_point]).item()

    return loss_val




def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def hg_propagate(new_g, tgt_type, num_hops, max_hops, extra_metapath, echo=False):
    for hop in range(1, max_hops):
        reserve_heads = [ele[:hop] for ele in extra_metapath if len(ele) > hop]
        for etype in new_g.etypes:
            stype, _, dtype = new_g.to_canonical_etype(etype)

            for k in list(new_g.nodes[stype].data.keys()):
                if len(k) == hop:
                    current_dst_name = f'{dtype}{k}'
                    if (hop == num_hops and dtype != tgt_type and k not in reserve_heads) \
                      or (hop > num_hops and k not in reserve_heads):
                        continue
                    if echo: print(k, etype, current_dst_name)
                    new_g[etype].update_all(
                        fn.copy_u(k, 'm'),
                        fn.mean('m', current_dst_name), etype=etype)

        # remove no-use items
        for ntype in new_g.ntypes:
            if ntype == tgt_type: continue
            removes = []
            for k in new_g.nodes[ntype].data.keys():
                if len(k) <= hop:
                    removes.append(k)
            for k in removes:
                new_g.nodes[ntype].data.pop(k)
            if echo and len(removes): print('remove', removes)
        gc.collect()

        if echo: print(f'-- hop={hop} ---')
        for ntype in new_g.ntypes:
            for k, v in new_g.nodes[ntype].data.items():
                if echo: print(f'{ntype} {k} {v.shape}')
        if echo: print(f'------\n')

    return new_g


def hg_propagate_search(new_g, tgt_type, num_hops, max_hops, extra_metapath, echo=False, prop_device='cpu'):
    for hop in range(1, max_hops):
        reserve_heads = [ele[:hop] for ele in extra_metapath if len(ele) > hop]
        for etype in new_g.etypes:
            stype, _, dtype = new_g.to_canonical_etype(etype)

            for k in list(new_g.nodes[stype].data.keys()):
                if len(k) == hop:
                    current_dst_name = f'{dtype}{k}'
                    if (hop == num_hops and dtype != tgt_type and k not in reserve_heads) \
                      or (hop > num_hops and k not in reserve_heads):
                        continue
                    if echo: print(k, etype, current_dst_name)
                    new_g[etype].update_all(
                        fn.copy_u(k, 'm'),
                        fn.mean('m', current_dst_name), etype=etype)

        # remove no-use items
        for ntype in new_g.ntypes:
            if ntype == tgt_type: continue
            removes = []
            for k in new_g.nodes[ntype].data.keys():
                if len(k) <= hop:
                    removes.append(k)
            for k in removes:
                new_g.nodes[ntype].data.pop(k)
            if echo and len(removes): print('remove', removes)
        gc.collect()

        if echo: print(f'-- hop={hop} ---')
        for ntype in new_g.ntypes:
            for k, v in new_g.nodes[ntype].data.items():
                if echo: print(f'{ntype} {k} {v.shape}')
        if echo: print(f'------\n')

    return new_g
# device = "cuda:{}".format(args.gpu) if not args.cpu else 'cpu'
# g = hg_propagate_split_on_gpu(g, tgt_type, args.num_hops, max_hops, extra_metapath,False,device,32)
def hg_propagate_split_on_gpu(new_g, tgt_type, num_hops, max_hops, extra_metapath,echo=False,device = 'cpu',split_num = 32, half=False):
    cpu_embedding = {}
    out_embedding = {}
    cpu_split = {}
    for ntype in new_g.ntypes:
        cpu_embedding[ntype] = new_g.nodes[ntype].data.pop(ntype).t()
        if half:
            cpu_embedding[ntype] = cpu_embedding[ntype].half()#.type(torch.float16)
            
        if split_num > cpu_embedding[ntype].shape[0]:
            raise KeyError
        split_len =  cpu_embedding[ntype].shape[0]//split_num
        this_split = [[int(split_len*i),int(split_len*(i+1))] for i in range(split_num)]
        this_split[-1][-1] = cpu_embedding[ntype].shape[0]
        cpu_split[ntype] = this_split

    new_g = new_g.to(device)

    for split_index in range(split_num):
        for ntype in new_g.ntypes:
            to_split = cpu_split[ntype][split_index]
            new_g.nodes[ntype].data[ntype] = cpu_embedding[ntype][to_split[0]:to_split[1]].t().to(device)

        
        for hop in range(1, max_hops):
            reserve_heads = [ele[:hop] for ele in extra_metapath if len(ele) > hop]
            for etype in new_g.etypes:
                stype, _, dtype = new_g.to_canonical_etype(etype)

                for k in list(new_g.nodes[stype].data.keys()):
                    if len(k) == hop:
                        current_dst_name = f'{dtype}{k}'
                        if (hop == num_hops and dtype != tgt_type and k not in reserve_heads) \
                        or (hop > num_hops and k not in reserve_heads):
                            continue
                        if echo: print(k, etype, current_dst_name)
                        new_g[etype].update_all(
                            fn.copy_u(k, 'm'),
                            fn.mean('m', current_dst_name), etype=etype)

            # remove no-use items
            for ntype in new_g.ntypes:
                if ntype == tgt_type: continue
                removes = []
                for k in new_g.nodes[ntype].data.keys():
                    if len(k) <= hop:
                        removes.append(k)
                for k in removes:
                    new_g.nodes[ntype].data.pop(k)
                if echo and len(removes): print('remove', removes)
            gc.collect()

            if echo: print(f'-- hop={hop} ---')
            for ntype in new_g.ntypes:
                for k, v in new_g.nodes[ntype].data.items():
                    if echo: print(f'{ntype} {k} {v.shape}')
            if echo: print(f'------\n')

        for key in list(new_g.nodes[tgt_type].data.keys()):
            if key not in out_embedding:
                out_embedding[key] = new_g.nodes[tgt_type].data.pop(key).cpu()
            else:
                out_embedding[key] = torch.cat([out_embedding[key],new_g.nodes[tgt_type].data.pop(key).cpu()],dim=1)

            # out_embedding[key].append()

        for ntype in new_g.ntypes:
            new_g.nodes[ntype].data.clear()
            # for k in new_g.nodes[ntype].data.keys():
            #     new_g.nodes[ntype].data.pop(k)
        gc.collect()

    new_g = new_g.to('cpu')
    for key in list(out_embedding.keys()):
        new_g.nodes[ntype].data[key] = out_embedding[key]

    return new_g

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

def find_index(arr,target):
    for index,item in enumerate(arr):
        if item ==target:
            return index
    return -1


def propagate_self_value(g,target_node = 'P',max_hop = 1):
    import scipy.sparse as sp
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
    for index in tqdm(range(total_node_num)):
        out_dict[target_node].append(1)
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

                    if new_str[-1] == target_node:
                        get = new_csr.indices.tolist()
                        real_index = find_index(get,index)
                        if real_index!= -1:
                            out_dict[new_str].append(new_csr.data[real_index])
                        else:
                            out_dict[new_str].append(0.0)
                        
        new_out_dict = {}
        for key in out_dict.keys():
            new_out_dict[key] = torch.tensor(out_dict[key])

    return new_out_dict


def propagate_self_value_parallel(g,target_node = 'P',max_hop = 1,parallel_num = 10):
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
    
    def the_func(index,out_name=False):
        my_name = []
        this_out = []
        my_name.append(target_node)
        this_out.append(1)
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

                    if new_str[-1] == target_node:
                        get = new_csr.indices.tolist()
                        real_index = find_index(get,index)
                        if real_index!= -1:
                            my_name.append(new_str)
                            this_out.append(new_csr.data[real_index])
                        else:
                            my_name.append(new_str)
                            this_out.append(0.0)

        if out_name:
            return my_name,this_out
        else:
            return this_out
        
    out_name,out_list = the_func(0,True)
    out_list = [out_list]
    extra_list = Parallel(n_jobs=parallel_num)(delayed(the_func)(index) for index in tqdm(range(1,total_node_num)))
       
    extra_list = out_list+extra_list

    extra_list = torch.tensor(extra_list)
    out_dir = {}
    for index,key in enumerate(out_name):
        out_dir[key] = extra_list[:,index]



    return out_dir


def propagate_self_value_gpu_parallel(g,target_node = 'P',max_hop = 1,parallel_num = 10,device='cuda:2'):
    import  scipy.sparse as sp
    from collections import defaultdict,deque
    import math

    my_adjs = {}
    in_links = defaultdict(list)
    for etype in g.etypes:
        sparse_coo = g.adj(etype=etype).coalesce()
        sparse_csr = sp.csr_matrix((sparse_coo.values(),sparse_coo.indices()),shape = sparse_coo.shape)
        sparse_csr = sp_normalize(sparse_csr)
        sparse_coo = torch.sparse_coo_tensor(torch.LongTensor(sparse_csr.nonzero()), torch.FloatTensor(sparse_csr.data),
                    size=sparse_csr.shape,device=device)
        my_adjs[etype] = sparse_coo
        in_links[etype[0]].append(etype)

    total_node_num = g.nodes[target_node].data[target_node].shape[0]

    out_dict = defaultdict(list)
    iter_num = int(math.ceil(total_node_num/parallel_num))
    for index in tqdm(range(iter_num)):
        start_num = index*parallel_num
        end_num = min((index+1)*parallel_num,total_node_num)
        index_list = list(range(start_num,end_num))
        index_num = len(index_list)
        out_dict[target_node].extend([1.0]*index_num)
        the_queue = deque()
        init_coo = torch.sparse_coo_tensor((index_list,index_list),[1.0]*index_num,size=(total_node_num,total_node_num),device=device)
        the_queue.append((target_node,init_coo))
        index_list = list(zip(index_list,index_list))
        while  len(the_queue) >0 :
            item = the_queue.popleft()
            if len(item[0]) <max_hop:
                for links in in_links[item[0][-1]]:
                    new_str = item[0]+links[-1]
                    new_coo = torch.sparse.mm(item[1],my_adjs[links])
                    the_queue.append((new_str,new_coo))

                    if new_str[-1] == target_node:
                        get = [new_coo[get_index].item() for get_index in index_list]
                        out_dict[new_str].extend(get)

            elif len(item[0]) == max_hop:
                for links in in_links[item[0][-1]]:
                    if links[-1] == target_node:
                        new_str = item[0]+links[-1]
                        new_coo = torch.sparse.mm(item[1],my_adjs[links])
                        get = [new_coo[get_index].item() for get_index in index_list]
                        out_dict[new_str].extend(get)

        new_out_dict = {}
        for key in out_dict.keys():
            new_out_dict[key] = torch.tensor(out_dict[key])


    return new_out_dict


def clear_hg(new_g, echo=False):
    if echo: print('Remove keys left after propagation')
    for ntype in new_g.ntypes:
        keys = list(new_g.nodes[ntype].data.keys())
        if len(keys):
            if echo: print(ntype, keys)
            for k in keys:
                new_g.nodes[ntype].data.pop(k)
    return new_g


def check_acc(preds_dict, condition, init_labels, train_nid, val_nid, test_nid):
    mask_train, mask_val, mask_test = [], [], []
    remove_label_keys = []
    na, nb, nc = len(train_nid), len(val_nid), len(test_nid)

    for k, v in preds_dict.items():
        pred = v.argmax(1)

        a, b, c = pred[train_nid] == init_labels[train_nid], \
                  pred[val_nid] == init_labels[val_nid], \
                  pred[test_nid] == init_labels[test_nid]
        ra, rb, rc = a.sum() / len(train_nid), b.sum() / len(val_nid), c.sum() / len(test_nid)

        vv = torch.log((v / (v.sum(1, keepdim=True) + 1e-6)).clamp(1e-6, 1-1e-6))
        la, lb, lc = F.nll_loss(vv[train_nid], init_labels[train_nid]), \
                     F.nll_loss(vv[val_nid], init_labels[val_nid]), \
                     F.nll_loss(vv[test_nid], init_labels[test_nid])

        if condition(ra, rb, rc, k):
            mask_train.append(a)
            mask_val.append(b)
            mask_test.append(c)
        else:
            remove_label_keys.append(k)
        print(k, ra, rb, rc, la, lb, lc, (ra/rb-1)*100, (ra/rc-1)*100, (1-la/lb)*100, (1-la/lc)*100)

    print(set(list(preds_dict.keys())) - set(remove_label_keys))
    print((torch.stack(mask_train, dim=0).sum(0) > 0).sum() / len(train_nid))
    print((torch.stack(mask_val, dim=0).sum(0) > 0).sum() / len(val_nid))
    print((torch.stack(mask_test, dim=0).sum(0) > 0).sum() / len(test_nid))
    return remove_label_keys


def train(model, train_loader, loss_fcn, optimizer, evaluator, device,
          feats, label_feats, labels_cuda, label_emb, mask=None, scalar=None):
    model.train()
    total_loss = 0
    iter_num = 0
    y_true, y_pred = [], []

    for batch in train_loader:
        batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
        batch_labels_feats = {k: x[batch].to(device) for k, x in label_feats.items()}
        # if mask is not None:
        #     batch_mask = {k: x[batch].to(device) for k, x in mask.items()}
        # else:
        #     batch_mask = None
        batch_label_emb = label_emb[batch].to(device)
        batch_y = labels_cuda[batch]

        optimizer.zero_grad()
        if scalar is not None:
            with torch.cuda.amp.autocast():
                output_att = model(batch_feats, batch_labels_feats, batch_label_emb)
                if isinstance(loss_fcn, nn.BCELoss):
                    output_att = torch.sigmoid(output_att)
                loss_train = loss_fcn(output_att, batch_y)
            scalar.scale(loss_train).backward()
            scalar.step(optimizer)
            scalar.update()
        else:
            output_att = model(batch_feats, batch_labels_feats, batch_label_emb)
            if isinstance(loss_fcn, nn.BCELoss):
                output_att = torch.sigmoid(output_att)
            L1 = loss_fcn(output_att, batch_y)
            loss_train = L1
            loss_train.backward()
            optimizer.step()

        y_true.append(batch_y.cpu().to(torch.long))
        if isinstance(loss_fcn, nn.BCELoss):
            y_pred.append((output_att.data.cpu() > 0).int())
        else:
            y_pred.append(output_att.argmax(dim=-1, keepdim=True).cpu())
        total_loss += loss_train.item()
        iter_num += 1
    loss = total_loss / iter_num
    acc = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
    return loss, acc


def train_search(model, train_loader, loss_fcn, optimizer_w, optimizer_a, val_loader, epoch_sampled, evaluator, device,
          feats, label_feats, labels_cuda, label_emb, mask=None, scalar=None):
    model.train()
    total_loss = 0
    iter_num = 0
    y_true, y_pred = [], []
    val_total_loss = 0
    val_y_true, val_y_pred = [], []
    ###################  optimize w  ##################
    for batch in train_loader:
        val_batch = next(iter(val_loader)).to(device)
        batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
        val_batch_feats = {k: x[val_batch].to(device) for k, x in feats.items()}
        batch_labels_feats = {k: x[batch].to(device) for k, x in label_feats.items()}
        val_batch_labels_feats = {k: x[val_batch].to(device) for k, x in label_feats.items()}
        batch_label_emb = label_emb[batch].to(device)
        val_batch_label_emb = label_emb[val_batch].to(device)
        batch_y = labels_cuda[batch]
        val_batch_y = labels_cuda[val_batch]
        ########################################val  update w
        optimizer_w.zero_grad()
        if scalar is not None:
            with torch.cuda.amp.autocast():
                output_att = model(batch_feats, epoch_sampled, batch_labels_feats, batch_label_emb)
                if isinstance(loss_fcn, nn.BCELoss):
                    output_att = torch.sigmoid(output_att)
                loss_train = loss_fcn(output_att, batch_y)
            scalar.scale(loss_train).backward(retain_graph=True)
            scalar.step(optimizer_w)
            scalar.update()
        else:
            output_att = model(batch_feats, epoch_sampled, batch_labels_feats, batch_label_emb)
            if isinstance(loss_fcn, nn.BCELoss):
                output_att = torch.sigmoid(output_att)
            L1 = loss_fcn(output_att, batch_y)
            loss_train = L1
            loss_train.backward(retain_graph=True)
            optimizer_w.step()

        ########################################val  update a
        optimizer_a.zero_grad()
        if scalar is not None:
            with torch.cuda.amp.autocast():
                val_output_att = model(val_batch_feats, epoch_sampled, val_batch_labels_feats, val_batch_label_emb)
                if isinstance(loss_fcn, nn.BCELoss):
                    val_output_att = torch.sigmoid(val_output_att)
                val_loss_train = loss_fcn(val_output_att, val_batch_y)
            scalar.scale(val_loss_train).backward(retain_graph=True)
            scalar.step(optimizer_a)
            scalar.update()

        else:
            val_output_att = model(val_batch_feats, epoch_sampled, val_batch_labels_feats, val_batch_label_emb)
            if isinstance(loss_fcn, nn.BCELoss):
                val_output_att = torch.sigmoid(val_output_att)
            L1 = loss_fcn(val_output_att, val_batch_y)
            val_loss_train = L1
            val_loss_train.backward()
            optimizer_a.step()
        ########################################
        y_true.append(batch_y.cpu().to(torch.long))
        val_y_true.append(val_batch_y.cpu().to(torch.long))
        if isinstance(loss_fcn, nn.BCELoss):
            y_pred.append((output_att.data.cpu() > 0).int())
            val_y_pred.append((val_output_att.data.cpu() > 0).int())
        else:
            y_pred.append(output_att.argmax(dim=-1, keepdim=True).cpu())
            val_y_pred.append(val_output_att.argmax(dim=-1, keepdim=True).cpu())
        total_loss += loss_train.item()
        val_total_loss += val_loss_train.item()
        iter_num += 1
    loss_train = total_loss / iter_num
    acc_train = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
    loss_val = val_total_loss / iter_num
    acc_val = evaluator(torch.cat(val_y_true, dim=0), torch.cat(val_y_pred, dim=0))
    return loss_train, loss_val, acc_train, acc_val


def train_multi_stage(model, train_loader, enhance_loader, loss_fcn, optimizer, evaluator, device,
                      feats, label_feats, labels, label_emb, predict_prob, gama, scalar=None):
    model.train()
    loss_fcn = nn.CrossEntropyLoss()
    y_true, y_pred = [], []
    total_loss = 0
    loss_l1, loss_l2 = 0., 0.
    iter_num = 0
    for idx_1, idx_2 in zip(train_loader, enhance_loader):
        idx = torch.cat((idx_1, idx_2), dim=0)
        L1_ratio = len(idx_1) * 1.0 / (len(idx_1) + len(idx_2))
        L2_ratio = len(idx_2) * 1.0 / (len(idx_1) + len(idx_2))

        batch_feats = {k: x[idx].to(device) for k, x in feats.items()}
        batch_labels_feats = {k: x[idx].to(device) for k, x in label_feats.items()}
        batch_label_emb = label_emb[idx].to(device)
        y = labels[idx_1].to(torch.long).to(device)
        extra_weight, extra_y = predict_prob[idx_2].max(dim=1)
        extra_weight = extra_weight.to(device)
        extra_y = extra_y.to(device)

        # import code
        # code.interact(local=locals())

        optimizer.zero_grad()
        if scalar is not None:
            with torch.cuda.amp.autocast():
                output_att = model(batch_feats, batch_labels_feats, batch_label_emb)
                L1 = loss_fcn(output_att[:len(idx_1)],  y)
                L2 = F.cross_entropy(output_att[len(idx_1):], extra_y, reduction='none')
                L2 = (L2 * extra_weight).sum() / len(idx_2)
                loss_train = L1_ratio * L1 + gama * L2_ratio * L2
            scalar.scale(loss_train).backward()
            scalar.step(optimizer)
            scalar.update()
        else:
            output_att = model(batch_feats, label_emb[idx].to(device))
            L1 = loss_fcn(output_att[:len(idx_1)],  y)
            L2 = F.cross_entropy(output_att[len(idx_1):], extra_y, reduction='none')
            L2 = (L2 * extra_weight).sum() / len(idx_2)
            # teacher_soft = predict_prob[idx_2].to(device)
            # teacher_prob = torch.max(teacher_soft, dim=1, keepdim=True)[0]
            # L3 = (teacher_prob*(teacher_soft*(torch.log(teacher_soft+1e-8)-torch.log_softmax(output_att[len(idx_1):], dim=1)))).sum(1).mean()*(len(idx_2)*1.0/(len(idx_1)+len(idx_2)))
            # loss = L1 + L3*gama
            loss_train = L1_ratio * L1 + gama * L2_ratio * L2
            loss_train.backward()
            optimizer.step()

        y_true.append(labels[idx_1].to(torch.long))
        y_pred.append(output_att[:len(idx_1)].argmax(dim=-1, keepdim=True).cpu())
        total_loss += loss_train.item()
        loss_l1 += L1.item()
        loss_l2 += L2.item()
        iter_num += 1

    print(loss_l1 / iter_num, loss_l2 / iter_num)
    loss = total_loss / iter_num
    approx_acc = evaluator(torch.cat(y_true, dim=0),torch.cat(y_pred, dim=0))
    return loss, approx_acc


def train_search(model, train_loader, loss_fcn, optimizer_w, optimizer_a, val_loader, epoch_sampled, evaluator, device,
          feats, label_feats, labels_cuda, label_emb, mask=None, scalar=None):
    model.train()
    total_loss = 0
    iter_num = 0
    y_true, y_pred = [], []
    val_total_loss = 0
    val_y_true, val_y_pred = [], []
    ###################  optimize w  ##################
    for batch in train_loader:
        val_batch = next(iter(val_loader)).to(device)
        batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
        val_batch_feats = {k: x[val_batch].to(device) for k, x in feats.items()}
        batch_labels_feats = {k: x[batch].to(device) for k, x in label_feats.items()}
        val_batch_labels_feats = {k: x[val_batch].to(device) for k, x in label_feats.items()}
        batch_label_emb = label_emb[batch].to(device)
        val_batch_label_emb = label_emb[val_batch].to(device)
        batch_y = labels_cuda[batch]
        val_batch_y = labels_cuda[val_batch]
        ########################################val  update w
        optimizer_w.zero_grad()
        if scalar is not None:
            with torch.cuda.amp.autocast():
                output_att = model(epoch_sampled, batch_feats, batch_labels_feats, batch_label_emb)
                if isinstance(loss_fcn, nn.BCELoss):
                    output_att = torch.sigmoid(output_att)
                loss_train = loss_fcn(output_att, batch_y)
            scalar.scale(loss_train).backward(retain_graph=True)
            scalar.step(optimizer_w)
            scalar.update()
        else:
            output_att = model(epoch_sampled, batch_feats, batch_labels_feats, batch_label_emb)
            if isinstance(loss_fcn, nn.BCELoss):
                output_att = torch.sigmoid(output_att)
            L1 = loss_fcn(output_att, batch_y)
            loss_train = L1
            loss_train.backward(retain_graph=True)
            optimizer_w.step()

        ########################################val  update a
        optimizer_a.zero_grad()
        if scalar is not None:
            with torch.cuda.amp.autocast():
                val_output_att = model(epoch_sampled, val_batch_feats, val_batch_labels_feats, val_batch_label_emb)
                if isinstance(loss_fcn, nn.BCELoss):
                    val_output_att = torch.sigmoid(val_output_att)
                val_loss_train = loss_fcn(val_output_att, val_batch_y)
            scalar.scale(val_loss_train).backward(retain_graph=True)
            scalar.step(optimizer_a)
            scalar.update()

        else:
            val_output_att = model(epoch_sampled, val_batch_feats, val_batch_labels_feats, val_batch_label_emb)
            if isinstance(loss_fcn, nn.BCELoss):
                val_output_att = torch.sigmoid(val_output_att)
            L1 = loss_fcn(val_output_att, val_batch_y)
            val_loss_train = L1
            val_loss_train.backward()
            optimizer_a.step()
        ########################################
        y_true.append(batch_y.cpu().to(torch.long))
        val_y_true.append(val_batch_y.cpu().to(torch.long))
        if isinstance(loss_fcn, nn.BCELoss):
            y_pred.append((output_att.data.cpu() > 0).int())
            val_y_pred.append((val_output_att.data.cpu() > 0).int())
        else:
            y_pred.append(output_att.argmax(dim=-1, keepdim=True).cpu())
            val_y_pred.append(val_output_att.argmax(dim=-1, keepdim=True).cpu())
        total_loss += loss_train.item()
        val_total_loss += val_loss_train.item()
        iter_num += 1
    loss_train = total_loss / iter_num
    acc_train = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
    loss_val = val_total_loss / iter_num
    acc_val = evaluator(torch.cat(val_y_true, dim=0), torch.cat(val_y_pred, dim=0))
    return loss_train, loss_val, acc_train, acc_val


@torch.no_grad()
def gen_output_torch(model, feats, label_feats, label_emb, test_loader, device):
    model.eval()
    preds = []
    for batch in tqdm(test_loader):
        batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
        batch_labels_feats = {k: x[batch].to(device) for k, x in label_feats.items()}
        batch_label_emb = label_emb[batch].to(device)
        preds.append(model(batch_feats, batch_labels_feats,batch_label_emb).cpu())
    preds = torch.cat(preds, dim=0)
    return preds


def gen_output_torch_search(model, feats, label_feats, label_emb, test_loader, device, idx):
    model.eval()
    preds = []
    for batch in tqdm(test_loader):
        batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
        batch_labels_feats = {k: x[batch].to(device) for k, x in label_feats.items()}
        batch_label_emb = label_emb[batch].to(device)
        preds.append(model(batch_feats, idx, batch_labels_feats, batch_label_emb).cpu())
    preds = torch.cat(preds, dim=0)
    return preds


def get_ogb_evaluator(dataset):
    evaluator = Evaluator(name=dataset)
    return lambda preds, labels: evaluator.eval({
            "y_true": labels.view(-1, 1),
            "y_pred": preds.view(-1, 1),
        })["acc"]


def load_dataset(args):
    if args.dataset == 'ogbn-mag':
        # train/val/test 629571/64879/41939
        return load_mag(args)
    else:
        assert 0, 'Only allowed [ogbn-mag]'


def degree_limit(edge_node,max_degree):
    from collections import Counter
    edge_node_list = edge_node.cpu().tolist()
    node_Counter = Counter(edge_node_list)
    out_Counter = {}
    for key in node_Counter.keys():
        if node_Counter[key] > max_degree:
            out_Counter[key] = max_degree


    out_mask = torch.zeros(edge_node.shape) != 0.0
    for index,key in enumerate(edge_node_list):
        if key not in out_Counter:
            out_mask[index] = True
        else:
            if out_Counter[key] >= 1:
                out_mask[index] = True
                out_Counter[key] -= 1

    return out_mask


def load_mag(args, symmetric=True):
    dataset = DglNodePropPredDataset(name=args.dataset, root=args.root)
    splitted_idx = dataset.get_idx_split()

    g, init_labels = dataset[0]
    splitted_idx = dataset.get_idx_split()
    train_nid = splitted_idx['train']['paper']
    val_nid = splitted_idx['valid']['paper']
    test_nid = splitted_idx['test']['paper']

    features = g.nodes['paper'].data['feat']
    if len(args.extra_embedding):
        print(f'Use extra embeddings generated with the {args.extra_embedding} method')
        # path = os.path.join(args.emb_path, f'{args.extra_embedding}_nars')
        path = args.emb_path

        author_emb = torch.load(os.path.join(path, 'author.pt'), map_location=torch.device('cpu')).float()
        topic_emb = torch.load(os.path.join(path, 'field_of_study.pt'), map_location=torch.device('cpu')).float()
        institution_emb = torch.load(os.path.join(path, 'institution.pt'), map_location=torch.device('cpu')).float()
    else:
        author_emb = torch.Tensor(g.num_nodes('author'), args.embed_size).uniform_(-0.5, 0.5)
        topic_emb = torch.Tensor(g.num_nodes('field_of_study'), args.embed_size).uniform_(-0.5, 0.5)
        institution_emb = torch.Tensor(g.num_nodes('institution'), args.embed_size).uniform_(-0.5, 0.5)

    g.nodes['paper'].data['feat'] = features
    g.nodes['author'].data['feat'] = author_emb
    g.nodes['institution'].data['feat'] = institution_emb
    g.nodes['field_of_study'].data['feat'] = topic_emb

    init_labels = init_labels['paper'].squeeze()
    n_classes = int(init_labels.max()) + 1
    evaluator = get_ogb_evaluator(args.dataset)

    # for k in g.ntypes:
    #     print(k, g.ndata['feat'][k].shape)
    for k in g.ntypes:
        print(k, g.nodes[k].data['feat'].shape)

    edge_mask_ratio = args.edge_mask_ratio

    adjs = []

    # generator = torch.Generator().manual_seed(args.mask_seed)
    out_max_degree = args.out_max_deg # 5
    in_max_degree = args.in_max_deg # 0

    num_edgs = 0

    for i, etype in enumerate(g.etypes):
        src, dst, eid = g._graph.edges(i)
        edge_keep = None

        if out_max_degree is not None and out_max_degree > 0:
            edge_keep = degree_limit(src,out_max_degree)
        elif in_max_degree is not None and in_max_degree > 0:
            edge_keep = degree_limit(dst,in_max_degree)


        if edge_keep != None:
            src = src[edge_keep]
            dst = dst[edge_keep]
            eid = eid[edge_keep]

        adj = SparseTensor(row=dst, col=src)
        adjs.append(adj)
        print(g.to_canonical_etype(etype), adj)
    
    # print("num_edgs:", num_edgs)
    # exit()

    # F --- *P --- A --- I
    # paper : [736389, 128]
    # author: [1134649, 256]
    # institution [8740, 256]
    # field_of_study [59965, 256]

    new_edges = {}
    ntypes = set()

    etypes = [ # src->tgt
        ('A', 'A-I', 'I'),
        ('A', 'A-P', 'P'),
        ('P', 'P-P', 'P'),
        ('P', 'P-F', 'F'),
    ]

    if symmetric:
        adjs[2] = adjs[2].to_symmetric()
        assert torch.all(adjs[2].get_diag() == 0)

    for etype, adj in zip(etypes, adjs):
        stype, rtype, dtype = etype
        dst, src, _ = adj.coo()
        src = src.numpy()
        dst = dst.numpy()
        if stype == dtype:
            new_edges[(stype, rtype, dtype)] = (np.concatenate((src, dst)), np.concatenate((dst, src)))
        else:
            new_edges[(stype, rtype, dtype)] = (src, dst)
            new_edges[(dtype, rtype[::-1], stype)] = (dst, src)
        ntypes.add(stype)
        ntypes.add(dtype)
    num_nodes_dict = {'P':g.nodes['paper'].data['feat'].shape[0],
                     'A':g.nodes['author'].data['feat'].shape[0],
                     'I':g.nodes['institution'].data['feat'].shape[0],
                     'F':g.nodes['field_of_study'].data['feat'].shape[0]}
    new_g = dgl.heterograph(new_edges,num_nodes_dict = num_nodes_dict)
    new_g.nodes['P'].data['P'] = g.nodes['paper'].data['feat']
    new_g.nodes['A'].data['A'] = g.nodes['author'].data['feat']
    new_g.nodes['I'].data['I'] = g.nodes['institution'].data['feat']
    new_g.nodes['F'].data['F'] = g.nodes['field_of_study'].data['feat']

    IA, PA, PP, FP = adjs

    diag_name = f'{args.dataset}_PFP_diag.pt'
    if not os.path.exists(diag_name):
        PF = FP.t()
        PFP_diag = sparse_tools.spspmm_diag_sym_ABA(PF)
        torch.save(PFP_diag, diag_name)

    if symmetric:
        diag_name = f'{args.dataset}_PPP_diag.pt'
        if not os.path.exists(diag_name):
            # PP = PP.to_symmetric()
            # assert torch.all(PP.get_diag() == 0)
            PPP_diag = sparse_tools.spspmm_diag_sym_AAA(PP)
            torch.save(PPP_diag, diag_name)
    else:
        assert False

    diag_name = f'{args.dataset}_PAP_diag.pt'
    if not os.path.exists(diag_name):
        PAP_diag = sparse_tools.spspmm_diag_sym_ABA(PA)
        torch.save(PAP_diag, diag_name)

    return new_g, init_labels, new_g.num_nodes('P'), n_classes, train_nid, val_nid, test_nid, evaluator

