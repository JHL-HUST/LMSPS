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

from model_search import *
from utils import *
from sparse_tools import SparseAdjList



def main(args):
    if args.seed > 0:
        set_random_seed(args.seed)

    g, adjs, init_labels, num_classes, dl, train_nid, val_nid, test_nid, test_nid_full \
        = load_dataset(args)

    for k in adjs.keys():
        adjs[k].storage._value = None
        adjs[k].storage._value = torch.ones(adjs[k].nnz()) / adjs[k].sum(dim=-1)[adjs[k].storage.row()]

    # =======
    # rearange node idx (for feats & labels)
    # =======
    search_node_nums= (len(train_nid)+len(val_nid))
    train_node_nums = search_node_nums // 2
    valid_node_nums = search_node_nums // 2
    test_node_nums = len(test_nid)
    trainval_point = train_node_nums
    valtest_point = trainval_point + valid_node_nums
    total_num_nodes = train_node_nums + valid_node_nums + test_node_nums


    num_nodes = dl.nodes['count'][0]

    if total_num_nodes < num_nodes:
        flag = np.ones(num_nodes, dtype=bool)
        flag[train_nid] = 0
        flag[val_nid] = 0
        flag[test_nid] = 0
        extra_nid = np.where(flag)[0]
        print(f'Find {len(extra_nid)} extra nid for dataset {args.dataset}')
    else:
        extra_nid = np.array([])

    init2sort = torch.LongTensor(np.concatenate([train_nid, val_nid, test_nid, extra_nid]))
    sort2init = torch.argsort(init2sort)
    assert torch.all(init_labels[init2sort][sort2init] == init_labels)
    labels = init_labels[init2sort]

    # =======
    # neighbor aggregation
    # =======
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
    elif args.dataset == 'Freebase':
        tgt_type = '0'
        node_types = [str(i) for i in range(8)]
        extra_metapath = []
    else:
        assert 0
    extra_metapath = [ele for ele in extra_metapath if len(ele) > args.num_hops + 1]

    print(f'Current num hops = {args.num_hops}')

    if args.dataset == 'Freebase':
        prop_device = 'cuda:{}'.format(args.gpu) if not args.cpu else 'cpu'
    else:
        prop_device = 'cpu'
    store_device = 'cpu'

    if args.dataset == 'Freebase':
        if not os.path.exists('./Freebase_adjs'):
            os.makedirs('./Freebase_adjs')
        num_tgt_nodes = dl.nodes['count'][0]

    # compute k-hop feature
    prop_tic = datetime.datetime.now()
    if args.dataset != 'Freebase':
        if len(extra_metapath):
            max_length = max(args.num_hops + 1, max([len(ele) for ele in extra_metapath]))
        else:
            max_length = args.num_hops + 1


        g = hg_propagate_feat_dgl(g, tgt_type, args.num_hops, max_length, echo=True)


        feats = {}
        keys = list(g.nodes[tgt_type].data.keys())
        print(f'For tgt {tgt_type}, feature keys {keys}')
        for k in keys:
            feats[k] = g.nodes[tgt_type].data.pop(k)

    else:
        if len(extra_metapath):
            max_length = max(args.num_hops + 1, max([len(ele) for ele in extra_metapath]))
        else:
            max_length = args.num_hops + 1

        save_name = f'./Freebase_adjs/feat_seed{args.seed}_hop{args.num_hops}'
        if args.seed > 0 and os.path.exists(f'{save_name}_00_int64.npy'):
            # meta_adjs = torch.load(save_name)
            meta_adjs = {}
            for srcname in tqdm(dl.nodes['count'].keys()):
                tmp = SparseAdjList(f'{save_name}_0{srcname}', None, None, num_tgt_nodes, dl.nodes['count'][srcname], with_values=True)
                for k in tmp.keys:
                    assert k not in meta_adjs
                meta_adjs.update(tmp.load_adjs(expand=True))
                del tmp
        else:
            meta_adjs = hg_propagate_sparse_pyg(adjs, tgt_type, args.num_hops, max_length, extra_metapath, prop_feats=True, echo=True, prop_device=prop_device)

            meta_adj_list = []
            for srcname in dl.nodes['count'].keys():
                keys = [k for k in meta_adjs.keys() if k[-1] == str(srcname)]
                tmp = SparseAdjList(f'{save_name}_0{srcname}', keys, meta_adjs, num_tgt_nodes, dl.nodes['count'][srcname], with_values=True)
                meta_adj_list.append(tmp)

            for srcname in dl.nodes['count'].keys():
                tmp = SparseAdjList(f'{save_name}_0{srcname}', None, None, num_tgt_nodes, dl.nodes['count'][srcname], with_values=True)
                tmp_adjs = tmp.load_adjs(expand=True)
                print(srcname, tmp.keys)
                for k in tmp.keys:
                    assert torch.all(meta_adjs[k].storage.rowptr() == tmp_adjs[k].storage.rowptr())
                    assert torch.all(meta_adjs[k].storage.col() == tmp_adjs[k].storage.col())
                    assert torch.all(meta_adjs[k].storage.value() == tmp_adjs[k].storage.value())
                del tmp_adjs, tmp
                gc.collect()

        feats = {k: v.clone() for k, v in meta_adjs.items() if len(k) <= args.num_hops + 1 or k in extra_metapath}
        assert '0' not in feats
        feats['0'] = SparseTensor.eye(dl.nodes['count'][0])
        print(f'For tgt {tgt_type}, Involved keys {feats.keys()}')

    if args.dataset in ['DBLP', 'ACM', 'IMDB']:
        data_size = {k: v.size(-1) for k, v in feats.items()}
        feats = {k: v[init2sort] for k, v in feats.items()}
    elif args.dataset == 'Freebase':
        data_size = dict(dl.nodes['count'])

        for k, v in tqdm(feats.items()):
            if len(k) == 1:
                continue

            if k[0] == '0' and k[-1] == '0':

                feats[k], _ = v.sample_adj(init2sort, -1, False) # faster, 50% time acceleration
            elif k[0] == '0':

                feats[k] = v[init2sort]
            else:
                assert args.two_layer, k
                if k[-1] == tgt_type:
                    feats[k] = v[:, init2sort]
                else:
                    feats[k] = v
    else:
        assert 0
    prop_toc = datetime.datetime.now()
    print(f'Time used for feat prop {prop_toc - prop_tic}')
    gc.collect()

    # =======
    checkpt_folder = f'./output/{args.dataset}/'
    if not os.path.exists(checkpt_folder):
        os.makedirs(checkpt_folder)
    checkpt_file = checkpt_folder + uuid.uuid4().hex
    #print('checkpt_file', checkpt_file)

    if args.amp:
        scalar = torch.cuda.amp.GradScaler()
    else:
        scalar = None

    device = 'cuda:{}'.format(args.gpu) if not args.cpu else 'cpu'
    if args.dataset != 'IMDB':
        labels_cuda = labels.long().to(device)
    else:
        labels = labels.float()
        labels_cuda = labels.to(device)

    for stage in [0]:
        epochs = args.stage

        # =======
        # labels propagate alongside the metapath
        # =======
        label_feats = {}
        if args.label_feats:
            if args.dataset != 'IMDB':
                label_onehot = torch.zeros((num_nodes, num_classes))
                label_onehot[train_nid] = F.one_hot(init_labels[train_nid], num_classes).float()
            else:
                label_onehot = torch.zeros((num_nodes, num_classes))
                label_onehot[train_nid] = init_labels[train_nid].float()

            if args.dataset == 'DBLP':
                extra_metapath = []
            elif args.dataset == 'IMDB':
                extra_metapath = []
            elif args.dataset == 'ACM':
                extra_metapath = []
            elif args.dataset == 'Freebase':
                extra_metapath = []
            else:
                assert 0

            extra_metapath = [ele for ele in extra_metapath if len(ele) > args.num_label_hops + 1]
            if len(extra_metapath):
                max_length = max(args.num_label_hops + 1, max([len(ele) for ele in extra_metapath]))
            else:
                max_length = args.num_label_hops + 1

            print(f'Current label-prop num hops = {args.num_label_hops}')
            # compute k-hop feature
            prop_tic = datetime.datetime.now()
            if args.dataset == 'Freebase' and args.num_label_hops <= args.num_hops and len(extra_metapath) == 0:
                meta_adjs = {k: v for k, v in meta_adjs.items() if k[-1] == '0' and len(k) < max_length}
            else:
                if args.dataset == 'Freebase':
                    save_name = f'./Freebase_adjs/label_seed{args.seed}_hop{args.num_label_hops}'
                    if args.seed > 0 and os.path.exists(f'{save_name}_int64.npy'):
                        meta_adj_list = SparseAdjList(save_name, None, None, num_tgt_nodes, num_tgt_nodes, with_values=True)
                        meta_adjs = meta_adj_list.load_adjs(expand=True)
                    else:
                        meta_adjs = hg_propagate_sparse_pyg(
                            adjs, tgt_type, args.num_label_hops, max_length, extra_metapath, prop_feats=False, echo=True, prop_device=prop_device)
                        meta_adj_list = SparseAdjList(save_name, meta_adjs.keys(), meta_adjs, num_tgt_nodes, num_tgt_nodes, with_values=True)

                        tmp = SparseAdjList(save_name, None, None, num_tgt_nodes, num_tgt_nodes, with_values=True)
                        tmp_adjs = tmp.load_adjs(expand=True)
                        for k in tmp.keys:
                            assert torch.all(meta_adjs[k].storage.rowptr() == tmp_adjs[k].storage.rowptr())
                            assert torch.all(meta_adjs[k].storage.col() == tmp_adjs[k].storage.col())
                            assert torch.all(meta_adjs[k].storage.value() == tmp_adjs[k].storage.value())
                        del tmp_adjs, tmp
                        gc.collect()
                else:
                    # try:
                    #     meta_adjs = torch.load(f'./cache/{args.dataset}_label_prop_hop{args.num_label_hops}.pt')
                    # except:
                    meta_adjs = hg_propagate_sparse_pyg(
                        adjs, tgt_type, args.num_label_hops, max_length, extra_metapath, prop_feats=False, echo=True, prop_device=prop_device)
                    # torch.save(meta_adjs, f'./cache/{args.dataset}_label_prop_hop{args.num_label_hops}.pt')

            if args.dataset == 'Freebase':
                if 0:
                    label_onehot_g = label_onehot.to(prop_device)
                    for k, v in tqdm(meta_adjs.items()):
                        if args.dataset != 'Freebase':
                            label_feats[k] = remove_diag(v) @ label_onehot
                        else:
                            label_feats[k] = (remove_diag(v).to(prop_device) @ label_onehot_g).to(store_device)

                    del label_onehot_g
                    torch.cuda.empty_cache()
                    gc.collect()

                    condition = lambda ra,rb,rc,k: rb > 0.2
                    check_acc(label_feats, condition, init_labels, train_nid, val_nid, test_nid, show_test=False)

                    left_keys = ['00', '000', '0000', '0010', '0030', '0040', '0050', '0060', '0070']
                    remove_keys = list(set(list(label_feats.keys())) - set(left_keys))
                    for k in remove_keys:
                        label_feats.pop(k)
                else:
                    left_keys = ['00', '000', '0000', '0010', '0030', '0040', '0050', '0060', '0070']
                    remove_keys = list(set(list(meta_adjs.keys())) - set(left_keys))
                    for k in remove_keys:
                        meta_adjs.pop(k)

                    label_onehot_g = label_onehot.to(prop_device)
                    for k, v in tqdm(meta_adjs.items()):
                        if args.dataset != 'Freebase':
                            label_feats[k] = remove_diag(v) @ label_onehot
                        else:
                            label_feats[k] = (remove_diag(v).to(prop_device) @ label_onehot_g).to(store_device)

                    del label_onehot_g
                    torch.cuda.empty_cache()
                    gc.collect()
            else:
                for k, v in tqdm(meta_adjs.items()):
                    if args.dataset != 'Freebase':
                        label_feats[k] = remove_diag(v) @ label_onehot
                    else:
                        label_feats[k] = (remove_diag(v).to(prop_device) @ label_onehot_g).to(store_device)
                gc.collect()

                if args.dataset == 'IMDB':
                    condition = lambda ra,rb,rc,k: True
                    check_acc(label_feats, condition, init_labels, train_nid, val_nid, test_nid, show_test=False, loss_type='bce')
                else:
                    condition = lambda ra,rb,rc,k: True
                    check_acc(label_feats, condition, init_labels, train_nid, val_nid, test_nid, show_test=True)
            print('Involved label keys', label_feats.keys())

            label_feats = {k: v[init2sort] for k,v in label_feats.items()}
            prop_toc = datetime.datetime.now()
            print(f'Time used for label prop {prop_toc - prop_tic}')

        # =======
        # Train & eval loaders
        # =======
        train_loader = torch.utils.data.DataLoader(
            torch.arange(valtest_point//2), batch_size=args.batch_size, shuffle=True, drop_last=False)

        val_loader = torch.utils.data.DataLoader(
            torch.arange(valtest_point//2, valtest_point), batch_size=args.batch_size, shuffle=True, drop_last=False)



        # =======
        # Mask & Smooth
        # =======
        with_mask = False

        eval_loader, full_loader = [], []
        batchsize = 2 * args.batch_size


        for batch_idx in range((total_num_nodes-1) // batchsize + 1):
            batch_start = batch_idx * batchsize
            batch_end = min(total_num_nodes, (batch_idx+1) * batchsize)
            batch = torch.arange(batch_start, batch_end)

            batch_feats = {k: x[batch_start:batch_end] for k, x in feats.items()}
            batch_labels_feats = {k: x[batch_start:batch_end] for k, x in label_feats.items()}
            if with_mask:
                # batch_mask = {k: x[batch_start:batch_end] for k, x in full_mask.items()}
                ...
            else:
                batch_mask = None
            eval_loader.append((batch, batch_feats, batch_labels_feats, batch_mask))

        for batch_idx in range((num_nodes-total_num_nodes-1) // batchsize + 1):
            batch_start = batch_idx * batchsize + total_num_nodes
            batch_end = min(num_nodes, (batch_idx+1) * batchsize + total_num_nodes)
            batch = torch.arange(batch_start, batch_end)

            batch_feats = {k: x[batch_start:batch_end] for k, x in feats.items()}
            batch_labels_feats = {k: x[batch_start:batch_end] for k, x in label_feats.items()}
            if with_mask:
                # batch_mask = {k: x[batch_start:batch_end] for k, x in full_mask.items()}
                ...
            else:
                batch_mask = None
            full_loader.append((batch, batch_feats, batch_labels_feats, batch_mask))


        if args.ns_linear:
            ns_ratio = 2 * args.num_hops / (math.e ** (0.6 * args.num_hops))
            if ns_ratio > 0.5:
                ns_ratio = 0.5
            args.ns = math.floor((len(feats) + len(label_feats)) * ns_ratio)

        if args.ns > (len(feats) + len(label_feats)):
            args.ns = (len(feats) + len(label_feats))

        print(f"num sampled :{args.ns}")

        # =======
        # Construct network
        # =======
        torch.cuda.empty_cache()
        gc.collect()

        print(data_size.keys(), feats.keys(), label_feats.keys())

        model = LHMLP_Se(args.hidden, num_classes, feats.keys(), label_feats.keys(), tgt_type,
            args.dropout, args.input_drop, device, args.num_final, args.residual, bns=args.bns, data_size=data_size, num_sampled=args.ns)

        print(model)

        model = model.to(device)



        if args.seed == args.seeds[0]:
            #print(model)
            print("# Params:", get_n_params(model))

        if args.dataset == 'IMDB':
            loss_fcn = nn.BCEWithLogitsLoss()
        else:
            loss_fcn = nn.CrossEntropyLoss()
        optimizer_w = torch.optim.Adam(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)

        optimizer_a = torch.optim.Adam(model.alphas(), lr=args.alr)  #,weight_decay=args.weight_decay

        best_epoch = -1
        best_val_loss = 1000000
        best_test_loss = 0
        best_val = (0,0)
        best_test = (0,0)
        val_loss_list, test_loss_list = [], []
        val_acc_list, test_acc_list = [], []
        actual_loss_list, actual_acc_list = [], []
        store_list = []
        best_pred = None
        count = 0

        train_times = []



        for epoch in tqdm(range(args.stage)):
            gc.collect()
            torch.cuda.synchronize()


            # determain eps for operation selection
            eps = 1 - epoch/(args.stage - 1)

            epoch_sampled = model.epoch_sample(eps)
            meta_path_sampled = [model.all_meta_path[i] for i in range(model.num_feats) if i in epoch_sampled]
            label_meta_path_sampled = [model.all_meta_path[i] for i in range(model.num_feats,model.num_paths) if i in epoch_sampled]

            epoch_feats = {k:v for k,v in feats.items() if k in meta_path_sampled or (model.residual and k==model.tgt_key)}  #
            epoch_label_feats = {k:v for k,v in label_feats.items() if k in label_meta_path_sampled}


            start = time.time()

            # determain tau
            model.set_tau(args.tau_max - (args.tau_max - args.tau_min) * epoch / (args.stage - 1))



            loss_w, loss_a, acc_w, acc_a = train_search_new(model, epoch_feats, epoch_label_feats, labels_cuda, loss_fcn, 
                                                            optimizer_w, optimizer_a, train_loader, val_loader,
                                                              epoch_sampled, meta_path_sampled, label_meta_path_sampled, 
                                                              evaluator, scalar=scalar)
            torch.cuda.synchronize()
            end = time.time()





            log = "" #"Epoch {}, training Time(s): {:.4f}, estimated train loss {:.4f}, acc {:.4f}, {:.4f}\n".format(epoch, end - start,loss, acc[0]*100, acc[1]*100)
            # print ("paths {}, label_path {}\n".format(path, label_path))

            torch.cuda.empty_cache()
            train_times.append(end-start)

        ##################direct evaluation##############
        # logging.info('Using valid loss crit for arch selection...')
        # model_new._initialize_flags()
        geno_out_vloss = project_op(list(feats.keys()), model, loss_fcn, eval_loader, device, trainval_point, valtest_point, labels, args.repeat)
        # return geno_out_vloss
        print('average train times', sum(train_times) / len(train_times))
        print("paths {}\n".format(geno_out_vloss))


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='SeHGNN')
    ## For environment costruction
    parser.add_argument("--seeds", nargs='+', type=int, default=[0],
                        help="the seed used in the training")
    parser.add_argument("--dataset", type=str, default="ogbn-mag")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action='store_true', default=False)
    parser.add_argument("--root", type=str, default="../data/")
    parser.add_argument("--stage", type=int, default=200, help="The epoch setting for each stage.")  # default 200
    parser.add_argument("--num-hops", type=int, default=2,
                        help="number of hops for propagation of raw labels")
    parser.add_argument("--label-feats", action='store_true', default=False,
                        help="whether to use the label propagated features")
    parser.add_argument("--num-label-hops", type=int, default=2,
                        help="number of hops for propagation of raw features")
    ## For network structure
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0,  # original 0.5
                        help="dropout on activation")
    parser.add_argument("--input-drop", type=float, default=0,  # original 0.1
                        help="input dropout of input features")
    parser.add_argument("--residual", action='store_true', default=False,
                        help="whether to add residual branch the raw input features")
    parser.add_argument("--bns", action='store_true', default=False,
                        help="whether to process the input features")
    ## for training
    parser.add_argument("--amp", action='store_true', default=False,
                        help="whether to amp to accelerate training with float16(half) calculation")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--alr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--patience", type=int, default=100,
                        help="early stop of times of the experiment")
    parser.add_argument("--drop-metapath", type=float, default=0,
                        help="whether to process the input features")
    ## for ablation
    parser.add_argument("--nh", type=int, default=1)
    parser.add_argument("--ns", type=int, default=30)
    parser.add_argument("--ratio", type=float, default=0)
    parser.add_argument("--dy", action='store_true', default=False)
    parser.add_argument("--no_path", nargs='+', type=str, default=[])
    parser.add_argument("--no_label", nargs='+', type=str, default=[])
    parser.add_argument("--identity", action='store_true', default=False)
    parser.add_argument("--topn", type=int, default=0)
    parser.add_argument("--all_path", action='store_true', default=False)
    parser.add_argument("--ns_linear", action='store_true', default=False)
    parser.add_argument('--tau_max', type=float, default=8, help='for gumbel softmax search gradient max value')
    parser.add_argument('--tau_min', type=float, default=4, help='for gumbel softmax search gradient min value')
    parser.add_argument("--edge_mask_ratio", type=float, default=0.0)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--num_final", type=int, default=60)
    parser.add_argument("--ACM_keep_F", action='store_true', default=False,
                        help="whether to use Field type")
    return parser.parse_args(args)

if __name__ == '__main__':
    args = parse_args()

    args.seed = args.seeds[0]
    print(args)


    for seed in args.seeds:
        args.seed = seed
        print('Restart with seed =', seed)
        main(args)

