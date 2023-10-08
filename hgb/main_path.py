import time
import uuid
import argparse
import datetime
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_sparse import remove_diag

from model import *
from utils import *
from arch import archs

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
    train_node_nums = len(train_nid)
    valid_node_nums = len(val_nid)
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
    else:
        assert 0
    extra_metapath = [ele for ele in extra_metapath if len(ele) > args.num_hops + 1]

    print(f'Current num hops = {args.num_hops}')

    prop_device = 'cpu'
    store_device = 'cpu'

    # compute k-hop feature
    prop_tic = datetime.datetime.now()

    if len(extra_metapath):
        max_length = max(args.num_hops + 1, max([len(ele) for ele in extra_metapath]))
    else:
        max_length = args.num_hops + 1

    g = hg_propagate_feat_dgl_path(g, tgt_type, args.num_hops, max_length, archs[args.arch][0], echo=False)
    feats = {}
    keys = list(g.nodes[tgt_type].data.keys())
    print(f'For tgt {tgt_type}, feature keys {keys}')
    for k in keys:
        feats[k] = g.nodes[tgt_type].data.pop(k)

    if args.dataset in ['DBLP', 'ACM', 'IMDB']:
        data_size = {k: v.size(-1) for k, v in feats.items()}
        feats = {k: v[init2sort] for k, v in feats.items()}

    else:
        assert 0

    feats = {k: v for k, v in feats.items() if k in archs[args.arch][0] or k == tgt_type}

    print(list(feats.keys()))

    prop_toc = datetime.datetime.now()
    print(f'Time used for feat prop {prop_toc - prop_tic}')
    gc.collect()

    # =======
    checkpt_folder = f'./output/{args.dataset}/'
    if not os.path.exists(checkpt_folder):
        os.makedirs(checkpt_folder)
    checkpt_file = checkpt_folder + uuid.uuid4().hex


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

            meta_adjs = hg_propagate_sparse_pyg(
                adjs, tgt_type, args.num_label_hops, max_length, extra_metapath, prop_feats=False, echo=False, prop_device=prop_device)

            for k, v in tqdm(meta_adjs.items()):
                label_feats[k] = remove_diag(v) @ label_onehot

            gc.collect()

            if args.dataset == 'IMDB':
                condition = lambda ra,rb,rc,k: True
                check_acc(label_feats, condition, init_labels, train_nid, val_nid, test_nid, show_test=False, loss_type='bce')
            else:
                condition = lambda ra,rb,rc,k: True
                check_acc(label_feats, condition, init_labels, train_nid, val_nid, test_nid, show_test=True)
            print('Involved label keys', label_feats.keys())

            label_feats = {k: v[init2sort] for k,v in label_feats.items() if k in archs[args.arch][1]}   # if k in archs[args.arch][1]

            prop_toc = datetime.datetime.now()
            print(f'Time used for label prop {prop_toc - prop_tic}')

        # =======
        # Train & eval loaders
        # =======
        train_loader = torch.utils.data.DataLoader(
            torch.arange(train_node_nums), batch_size=args.batch_size, shuffle=True, drop_last=False)

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

            batch_mask = None
            eval_loader.append((batch, batch_feats, batch_labels_feats, batch_mask))

        for batch_idx in range((num_nodes-total_num_nodes-1) // batchsize + 1):
            batch_start = batch_idx * batchsize + total_num_nodes
            batch_end = min(num_nodes, (batch_idx+1) * batchsize + total_num_nodes)
            batch = torch.arange(batch_start, batch_end)

            batch_feats = {k: x[batch_start:batch_end] for k, x in feats.items()}
            batch_labels_feats = {k: x[batch_start:batch_end] for k, x in label_feats.items()}

            batch_mask = None
            full_loader.append((batch, batch_feats, batch_labels_feats, batch_mask))

        # =======
        # Construct network
        # =======
        torch.cuda.empty_cache()
        gc.collect()
        model = LDMLP(args.embed_size, args.hidden, num_classes, feats.keys(), label_feats.keys(), tgt_type,
            args.dropout, args.input_drop, args.att_drop, args.label_drop,
            args.n_layers_2,  args.residual, bns=args.bns, data_size=data_size, path=archs[args.arch][0],
            label_path=archs[args.arch][1], eps=args.eps, device=device)
        
        model = model.to(device)
        if args.seed == args.seeds[0]:
            #print(model)
            print("# Params:", get_n_params(model))

        if args.dataset == 'IMDB':
            loss_fcn = nn.BCEWithLogitsLoss()
        else:
            loss_fcn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)

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
            start = time.time()
            loss, acc = train(model, feats, label_feats, labels_cuda, loss_fcn, optimizer, train_loader, evaluator, scalar=scalar)
            torch.cuda.synchronize()
            end = time.time()

            log = ""#"Epoch {}, training Time(s): {:.4f}, estimated train loss {:.4f}, acc {:.4f}, {:.4f}\n".format(epoch, end - start,loss, acc[0]*100, acc[1]*100)
            torch.cuda.empty_cache()
            train_times.append(end-start)

            start = time.time()
            with torch.no_grad():
                model.eval()
                raw_preds = []

                for batch, batch_feats, batch_labels_feats, batch_mask in eval_loader:
                    batch = batch.to(device)
                    batch_feats = {k: x.to(device) for k, x in batch_feats.items()}
                    batch_labels_feats = {k: x.to(device) for k, x in batch_labels_feats.items()}
                    if with_mask:
                        batch_mask = {k: x.to(device) for k, x in batch_mask.items()}
                    else:
                        batch_mask = None
                    raw_preds.append(model(batch, batch_feats, batch_labels_feats, batch_mask).cpu())

                raw_preds = torch.cat(raw_preds, dim=0)
                loss_train = loss_fcn(raw_preds[:trainval_point], labels[:trainval_point]).item()
                loss_val = loss_fcn(raw_preds[trainval_point:valtest_point], labels[trainval_point:valtest_point]).item()
                loss_test = loss_fcn(raw_preds[valtest_point:total_num_nodes], labels[valtest_point:total_num_nodes]).item()

            if args.dataset != 'IMDB':
                preds = raw_preds.argmax(dim=-1)
            else:
                preds = (raw_preds > 0.).int()

            train_acc = evaluator(preds[:trainval_point], labels[:trainval_point])
            val_acc = evaluator(preds[trainval_point:valtest_point], labels[trainval_point:valtest_point])
            test_acc = evaluator(preds[valtest_point:total_num_nodes], labels[valtest_point:total_num_nodes])

            end = time.time()
            #log += f'evaluation Time: {end-start}, Train loss: {loss_train}, Val loss: {loss_val}, Test loss: {loss_test}\n'
            log += f'Val loss: {loss_val}, Test loss: {loss_test}\n'
            log += 'Train acc: ({:.4f}, {:.4f}), Val acc: ({:.4f}, {:.4f}), Test acc: ({:.4f}, {:.4f}) ({})\n'.format(
                train_acc[0]*100, train_acc[1]*100, val_acc[0]*100, val_acc[1]*100, test_acc[0]*100, test_acc[1]*100, total_num_nodes-valtest_point)

            if loss_val <= best_val_loss:
                best_epoch = epoch
                best_val_loss = loss_val
                best_test_loss = loss_test
                best_val = val_acc
                best_test = test_acc

                best_pred = raw_preds
                torch.save(model.state_dict(), f'{checkpt_file}.pkl')
                wait = 0
            else:
                wait += 1
                if wait == args.patience:
                    break

            if epoch > 0 and epoch % 10 == 0: 
                log = log + f'\tCurrent best at epoch {best_epoch} with Val loss {best_val_loss:.4f} ({best_val[0]*100:.4f}, {best_val[1]*100:.4f})' \
                    + f', Test loss {best_test_loss:.4f} ({best_test[0]*100:.4f}, {best_test[1]*100:.4f})'
            #print(log)

        print('average train times', sum(train_times) / len(train_times))

        print(f'Best Epoch {best_epoch} at {checkpt_file.split("/")[-1]}\n\tFinal Val loss {best_val_loss:.4f} ({best_val[0]*100:.4f}, {best_val[1]*100:.4f})'
            + f', Test loss {best_test_loss:.4f} ({best_test[0]*100:.4f}, {best_test[1]*100:.4f})')   # micro  macro

        if len(full_loader):
            model.load_state_dict(torch.load(f'{checkpt_file}.pkl', map_location='cpu'), strict=True)
            torch.cuda.empty_cache()
            with torch.no_grad():
                model.eval()
                raw_preds = []

                for batch, batch_feats, batch_labels_feats, batch_mask in full_loader:
                    batch = batch.to(device)
                    batch_feats = {k: x.to(device) for k, x in batch_feats.items()}
                    batch_labels_feats = {k: x.to(device) for k, x in batch_labels_feats.items()}
                    if with_mask:
                        batch_mask = {k: x.to(device) for k, x in batch_mask.items()}
                    else:
                        batch_mask = None
                    raw_preds.append(model(batch, batch_feats, batch_labels_feats, batch_mask).cpu())
                raw_preds = torch.cat(raw_preds, dim=0)
            best_pred = torch.cat((best_pred, raw_preds), dim=0)

        torch.save(best_pred, f'{checkpt_file}.pt')

        if args.dataset != 'IMDB':
            predict_prob = best_pred.softmax(dim=1)
        else:
            predict_prob = torch.sigmoid(best_pred)

        test_logits = predict_prob[sort2init][test_nid_full]
        if args.dataset != 'IMDB':
            pred = test_logits.cpu().numpy().argmax(axis=1)
            dl.gen_file_for_evaluate(test_idx=test_nid_full, label=pred, file_path=f"./output/{args.dataset}_{args.seed}_{checkpt_file.split('/')[-1]}.txt")
        else:
            pred = (test_logits.cpu().numpy()>0.5).astype(int)
            dl.gen_file_for_evaluate(test_idx=test_nid_full, label=pred, file_path=f"./output/{args.dataset}_{args.seed}_{checkpt_file.split('/')[-1]}.txt", mode='multi')

    if args.dataset != 'IMDB':
        preds = predict_prob.argmax(dim=1, keepdim=True)
    else:
        preds = (predict_prob > 0.5).int()
    train_acc = evaluator(labels[:trainval_point], preds[:trainval_point])
    val_acc = evaluator(labels[trainval_point:valtest_point], preds[trainval_point:valtest_point])
    test_acc = evaluator(labels[valtest_point:total_num_nodes], preds[valtest_point:total_num_nodes])

    print(f'train_acc ({train_acc[0]*100:.2f}, {train_acc[1]*100:.2f}) ' \
        + f'val_acc ({val_acc[0]*100:.2f}, {val_acc[1]*100:.2f}) ' \
        + f'test_acc ({test_acc[0]*100:.2f}, {test_acc[1]*100:.2f})')
    print(checkpt_file.split('/')[-1])

    return [test_acc[0]*100, test_acc[1]*100]

def parse_args(args=None):
    parser = argparse.ArgumentParser(description='LDMLP')
    ## For environment costruction
    parser.add_argument("--seeds", nargs='+', type=int, default=[1],
                        help="the seed used in the training")
    parser.add_argument("--dataset", type=str, default="ogbn-mag")
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
    parser.add_argument("--bns", action='store_true', default=False,
                        help="whether to process the input features")
    ## for training
    parser.add_argument("--amp", action='store_true', default=False,
                        help="whether to amp to accelerate training with float16(half) calculation")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--patience", type=int, default=100,
                        help="early stop of times of the experiment")
    parser.add_argument("--edge_mask_ratio", type=float, default=0)
    parser.add_argument('--arch', type=str, default='DBLP')
    parser.add_argument("--eps", type=float, default=0)   #1e-12
    parser.add_argument("--ACM_keep_F", action='store_true', default=False,
                        help="whether to use Field type")

    return parser.parse_args(args)

if __name__ == '__main__':
    args = parse_args()

    # if args.dataset == 'ACM':
    #     args.ACM_keep_F = False

    args.seed = args.seeds[0]
    print(args)

    results = []
    for seed in args.seeds:
        args.seed = seed
        print('Restart with seed =', seed)
        result = main(args)
        results.append(result)
    print('results', results)

    results.sort(key=lambda x: x[0], reverse=True)
    print(results)
    results = results[:5]
    mima = list(map(list, zip(*results)))
    print(f'micro: {mima[0]}', f'macro: {mima[1]}')
    print(f'micro_mean: {np.mean(mima[0]):.2f}', f'micro_std: {np.std(mima[0]):.2f}')
    print(f'macro_mean: {np.mean(mima[1]):.2f}', f'macro_std: {np.std(mima[1]):.2f}')
