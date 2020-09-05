""" train.py

    Main interface to train the GNNs that will be later explained.
"""
import argparse
import os
import pickle
import random
import shutil
import time

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import networkx as nx
import numpy as np
import sklearn.metrics as metrics

import torch
import torch.nn as nn
from torch.autograd import Variable

from tensorboardX import SummaryWriter

import configs
import gengraph

import utils.math_utils as math_utils
import utils.io_utils as io_utils
import utils.parser_utils as parser_utils
import utils.train_utils as train_utils
import utils.featgen as featgen
import utils.graph_utils as graph_utils

import models


#############################
#
# Prepare Data
#
#############################
def prepare_data(graphs, args, test_graphs=None, max_nodes=0):

    random.shuffle(graphs)
    if test_graphs is None:
        train_idx = int(len(graphs) * args.train_ratio)
        test_idx = int(len(graphs) * (1 - args.test_ratio))
        train_graphs = graphs[:train_idx]
        val_graphs = graphs[train_idx:test_idx]
        test_graphs = graphs[test_idx:]
    else:
        train_idx = int(len(graphs) * args.train_ratio)
        train_graphs = graphs[:train_idx]
        val_graphs = graph[train_idx:]
    print(
        "Num training graphs: ",
        len(train_graphs),
        "; Num validation graphs: ",
        len(val_graphs),
        "; Num testing graphs: ",
        len(test_graphs),
    )

    print("Number of graphs: ", len(graphs))
    print("Number of edges: ", sum([G.number_of_edges() for G in graphs]))
    print(
        "Max, avg, std of graph size: ",
        max([G.number_of_nodes() for G in graphs]),
        ", " "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])),
        ", " "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])),
    )

    # minibatch
    dataset_sampler = graph_utils.GraphSampler(
        train_graphs,
        normalize=False,
        max_num_nodes=max_nodes,
        features=args.feature_type,
    )
    train_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    dataset_sampler = graph_utils.GraphSampler(
        val_graphs, 
        normalize=False, 
        max_num_nodes=max_nodes, 
        features=args.feature_type
    )
    val_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    dataset_sampler = graph_utils.GraphSampler(
        test_graphs,
        normalize=False,
        max_num_nodes=max_nodes,
        features=args.feature_type,
    )
    test_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    return (
        train_dataset_loader,
        val_dataset_loader,
        test_dataset_loader,
        dataset_sampler.max_num_nodes,
        dataset_sampler.feat_dim,
        dataset_sampler.assign_feat_dim,
    )


#############################
#
# Training 
#
#############################
def train(
    dataset,
    model,
    args,
    same_feat=True,
    val_dataset=None,
    test_dataset=None,
    writer=None,
    mask_nodes=True,
):
    writer_batch_idx = [0, 3, 6, 9]

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.001
    )
    iter = 0
    best_val_result = {"epoch": 0, "loss": 0, "acc": 0}
    test_result = {"epoch": 0, "loss": 0, "acc": 0}
    train_accs = []
    train_epochs = []
    best_val_accs = []
    best_val_epochs = []
    test_accs = []
    test_epochs = []
    val_accs = []

    for epoch in range(args.num_epochs):
        begin_time = time.time()
        avg_loss = 0.0
        model.train()
        predictions = []
        print("Epoch: ", epoch)
        for batch_idx, data in enumerate(dataset):
            model.zero_grad()
            if batch_idx == 0:
                prev_adjs = data["adj"]
                prev_feats = data["feats"]
                prev_labels = data["label"]
                all_adjs = prev_adjs
                all_feats = prev_feats
                all_labels = prev_labels
            elif batch_idx < 20:
                prev_adjs = data["adj"]
                prev_feats = data["feats"]
                prev_labels = data["label"]
                all_adjs = torch.cat((all_adjs, prev_adjs), dim=0)
                all_feats = torch.cat((all_feats, prev_feats), dim=0)
                all_labels = torch.cat((all_labels, prev_labels), dim=0)
            adj = Variable(data["adj"].float(), requires_grad=False).cuda()
            h0 = Variable(data["feats"].float(), requires_grad=False).cuda()
            label = Variable(data["label"].long()).cuda()
            batch_num_nodes = data["num_nodes"].int().numpy() if mask_nodes else None
            assign_input = Variable(
                data["assign_feats"].float(), requires_grad=False
            ).cuda()

            ypred, att_adj = model(h0, adj, batch_num_nodes, assign_x=assign_input)
            if batch_idx < 5:
                predictions += ypred.cpu().detach().numpy().tolist()

            if not args.method == "soft-assign" or not args.linkpred:
                loss = model.loss(ypred, label)
            else:
                loss = model.loss(ypred, label, adj, batch_num_nodes)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()
            iter += 1
            avg_loss += loss

        avg_loss /= batch_idx + 1
        elapsed = time.time() - begin_time
        if writer is not None:
            writer.add_scalar("loss/avg_loss", avg_loss, epoch)
            if args.linkpred:
                writer.add_scalar("loss/linkpred_loss", model.link_loss, epoch)
        print("Avg loss: ", avg_loss, "; epoch time: ", elapsed)
        result = evaluate(dataset, model, args, name="Train", max_num_examples=100)
        train_accs.append(result["acc"])
        train_epochs.append(epoch)
        if val_dataset is not None:
            val_result = evaluate(val_dataset, model, args, name="Validation")
            val_accs.append(val_result["acc"])
        if val_result["acc"] > best_val_result["acc"] - 1e-7:
            best_val_result["acc"] = val_result["acc"]
            best_val_result["epoch"] = epoch
            best_val_result["loss"] = avg_loss
        if test_dataset is not None:
            test_result = evaluate(test_dataset, model, args, name="Test")
            test_result["epoch"] = epoch
        if writer is not None:
            writer.add_scalar("acc/train_acc", result["acc"], epoch)
            writer.add_scalar("acc/val_acc", val_result["acc"], epoch)
            writer.add_scalar("loss/best_val_loss", best_val_result["loss"], epoch)
            if test_dataset is not None:
                writer.add_scalar("acc/test_acc", test_result["acc"], epoch)

        print("Best val result: ", best_val_result)
        best_val_epochs.append(best_val_result["epoch"])
        best_val_accs.append(best_val_result["acc"])
        if test_dataset is not None:
            print("Test result: ", test_result)
            test_epochs.append(test_result["epoch"])
            test_accs.append(test_result["acc"])

    matplotlib.style.use("seaborn")
    plt.switch_backend("agg")
    plt.figure()
    plt.plot(train_epochs, math_utils.exp_moving_avg(train_accs, 0.85), "-", lw=1)
    if test_dataset is not None:
        plt.plot(best_val_epochs, best_val_accs, "bo", test_epochs, test_accs, "go")
        plt.legend(["train", "val", "test"])
    else:
        plt.plot(best_val_epochs, best_val_accs, "bo")
        plt.legend(["train", "val"])
    plt.savefig(io_utils.gen_train_plt_name(args), dpi=600)
    plt.close()
    matplotlib.style.use("default")

    print(all_adjs.shape, all_feats.shape, all_labels.shape)

    cg_data = {
        "adj": all_adjs,
        "feat": all_feats,
        "label": all_labels,
        "pred": np.expand_dims(predictions, axis=0),
        "train_idx": list(range(len(dataset))),
    }
    io_utils.save_checkpoint(model, optimizer, args, num_epochs=-1, cg_dict=cg_data)
    return model, val_accs


def train_node_classifier(G, labels, model, args, writer=None):
    # train/test split only for nodes
    num_nodes = G.number_of_nodes()
    num_train = int(num_nodes * args.train_ratio)
    idx = [i for i in range(num_nodes)]

    np.random.shuffle(idx)
    train_idx = idx[:num_train]
    test_idx = idx[num_train:]

    data = gengraph.preprocess_input_graph(G, labels)
    labels_train = torch.tensor(data["labels"][:, train_idx], dtype=torch.long)
    adj = torch.tensor(data["adj"], dtype=torch.float)
    x = torch.tensor(data["feat"], requires_grad=True, dtype=torch.float)
    scheduler, optimizer = train_utils.build_optimizer(
        args, model.parameters(), weight_decay=args.weight_decay
    )
    model.train()
    ypred = None
    for epoch in range(args.num_epochs):
        begin_time = time.time()
        model.zero_grad()

        if args.gpu:
            ypred, adj_att = model(x.cuda(), adj.cuda())
        else:
            ypred, adj_att = model(x, adj)
        ypred_train = ypred[:, train_idx, :]
        if args.gpu:
            loss = model.loss(ypred_train, labels_train.cuda())
        else:
            loss = model.loss(ypred_train, labels_train)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()
        elapsed = time.time() - begin_time

        mask1 = torch.load("./workspace/feat_nodes_p={:.2f}.pkl".format(args.p))
        mask0 = [False if m else True for m in mask1]
        f0, f1 = evaluate_node(ypred.cpu(), data["labels"], mask0, mask1)
        if scheduler is not None:
            scheduler.step()

    torch.save(torch.max(ypred.cpu(),2)[1][0],"./workspace/GNN-{0}-prediction.pkl".format(args.name_suffix))
    TN0 = f0["conf_mat"][0][0]; FN0 = f0["conf_mat"][1][0]; TP0 = f0["conf_mat"][1][1]; FP0 = f0["conf_mat"][0][1]
    torch.save([TP0,FP0,TN0,FN0],"./workspace/GNN-{0}-f=0-performance.tmp".format(args.name_suffix))
    TN1 = f1["conf_mat"][0][0]; FN1 = f1["conf_mat"][1][0]; TP1 = f1["conf_mat"][1][1]; FP1 = f1["conf_mat"][0][1]
    torch.save([TP1,FP1,TN1,FN1],"./workspace/GNN-{0}-f=1-performance.tmp".format(args.name_suffix))
    torch.save([TP0+TP1,FP0+FP1,TN0+TN1,FN0+FN1],"./workspace/GNN-{0}-f=2-performance.tmp".format(args.name_suffix))
    # print("=================== GNN-{0} Result ===================".format(args.name_suffix))
    # print("| TP: {:7d} | FP: {:7d} | TN: {:7d} | FN: {:7d} |".format(TP,FP,TN,FN))
    # print("---------------------------------------------------------")
    # print("| Acc: {:.4f} | Pre: {:.4f} | TPR: {:.4f} | FPR: {:.4f} |".format((TP+TN)/(TP+FP+TN+FN) if (TP+FP+TN+FN) else 0.0,
    #                                                                          TP/(TP+FP) if (TP+FP) else 0.0,
    #                                                                          TP/(TP+FN) if (TP+FN) else 0.0,
    #                                                                          FP/(TN+FP) if (TN+FP) else 0.0))
    # print("========================================================="); print("")

    # computation graph
    model.eval()
    if args.gpu:
        ypred, _ = model(x.cuda(), adj.cuda())
    else:
        ypred, _ = model(x, adj)
    cg_data = {
        "adj": data["adj"],
        "feat": data["feat"],
        "label": data["labels"],
        "pred": ypred.cpu().detach().numpy(),
        "train_idx": train_idx,
    }
    # import pdb
    # pdb.set_trace()
    io_utils.save_checkpoint(model, optimizer, args, num_epochs=-1, cg_dict=cg_data)


def train_node_classifier_multigraph(G_list, labels, model, args, writer=None):
    train_idx_all, test_idx_all = [], []
    # train/test split only for nodes
    num_nodes = G_list[0].number_of_nodes()
    num_train = int(num_nodes * args.train_ratio)
    idx = [i for i in range(num_nodes)]
    np.random.shuffle(idx)
    train_idx = idx[:num_train]
    train_idx_all.append(train_idx)
    test_idx = idx[num_train:]
    test_idx_all.append(test_idx)

    data = gengraph.preprocess_input_graph(G_list[0], labels[0])
    all_labels = data["labels"]
    labels_train = torch.tensor(data["labels"][:, train_idx], dtype=torch.long)
    adj = torch.tensor(data["adj"], dtype=torch.float)
    x = torch.tensor(data["feat"], requires_grad=True, dtype=torch.float)

    for i in range(1, len(G_list)):
        np.random.shuffle(idx)
        train_idx = idx[:num_train]
        train_idx_all.append(train_idx)
        test_idx = idx[num_train:]
        test_idx_all.append(test_idx)
        data = gengraph.preprocess_input_graph(G_list[i], labels[i])
        all_labels = np.concatenate((all_labels, data["labels"]), axis=0)
        labels_train = torch.cat(
            [
                labels_train,
                torch.tensor(data["labels"][:, train_idx], dtype=torch.long),
            ],
            dim=0,
        )
        adj = torch.cat([adj, torch.tensor(data["adj"], dtype=torch.float)])
        x = torch.cat(
            [x, torch.tensor(data["feat"], requires_grad=True, dtype=torch.float)]
        )

    scheduler, optimizer = train_utils.build_optimizer(
        args, model.parameters(), weight_decay=args.weight_decay
    )
    model.train()
    ypred = None
    for epoch in range(args.num_epochs):
        begin_time = time.time()
        model.zero_grad()

        if args.gpu:
            ypred = model(x.cuda(), adj.cuda())
        else:
            ypred = model(x, adj)
        # normal indexing
        ypred_train = ypred[:, train_idx, :]
        # in multigraph setting we can't directly access all dimensions so we need to gather all the training instances
        all_train_idx = [item for sublist in train_idx_all for item in sublist]
        ypred_train_cmp = torch.cat(
            [ypred[i, train_idx_all[i], :] for i in range(10)], dim=0
        ).reshape(10, 146, 6)
        if args.gpu:
            loss = model.loss(ypred_train_cmp, labels_train.cuda())
        else:
            loss = model.loss(ypred_train_cmp, labels_train)
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.clip)

        optimizer.step()
        #for param_group in optimizer.param_groups:
        #    print(param_group["lr"])
        elapsed = time.time() - begin_time

        result_train, result_test = evaluate_node(
            ypred.cpu(), all_labels, train_idx_all, test_idx_all
        )
        if writer is not None:
            writer.add_scalar("loss/avg_loss", loss, epoch)
            writer.add_scalars(
                "prec",
                {"train": result_train["prec"], "test": result_test["prec"]},
                epoch,
            )
            writer.add_scalars(
                "recall",
                {"train": result_train["recall"], "test": result_test["recall"]},
                epoch,
            )
            writer.add_scalars(
                "acc", {"train": result_train["acc"], "test": result_test["acc"]}, epoch
            )

        print(
            "epoch: ",
            epoch,
            "; loss: ",
            loss.item(),
            "; train_acc: ",
            result_train["acc"],
            "; test_acc: ",
            result_test["acc"],
            "; epoch time: ",
            "{0:0.2f}".format(elapsed),
        )

        if scheduler is not None:
            scheduler.step()

    # computation graph
    model.eval()
    if args.gpu:
        ypred = model(x.cuda(), adj.cuda())
    else:
        ypred = model(x, adj)
    cg_data = {
        "adj": adj.cpu().detach().numpy(),
        "feat": x.cpu().detach().numpy(),
        "label": all_labels,
        "pred": ypred.cpu().detach().numpy(),
        "train_idx": train_idx_all,
    }
    io_utils.save_checkpoint(model, optimizer, args, num_epochs=-1, cg_dict=cg_data)



#############################
#
# Evaluate Trained Model
#
#############################
def evaluate(dataset, model, args, name="Validation", max_num_examples=None):
    model.eval()

    labels = []
    preds = []
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data["adj"].float(), requires_grad=False).cuda()
        h0 = Variable(data["feats"].float()).cuda()
        labels.append(data["label"].long().numpy())
        batch_num_nodes = data["num_nodes"].int().numpy()
        assign_input = Variable(
            data["assign_feats"].float(), requires_grad=False
        ).cuda()

        ypred, att_adj = model(h0, adj, batch_num_nodes, assign_x=assign_input)
        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())

        if max_num_examples is not None:
            if (batch_idx + 1) * args.batch_size > max_num_examples:
                break

    labels = np.hstack(labels)
    preds = np.hstack(preds)

    result = {
        "prec": metrics.precision_score(labels, preds, average="macro"),
        "recall": metrics.recall_score(labels, preds, average="macro"),
        "acc": metrics.accuracy_score(labels, preds),
    }
    print(name, " accuracy:", result["acc"])
    return result


def evaluate_node(ypred, labels, train_idx, test_idx):
    _, pred_labels = torch.max(ypred, 2)
    pred_labels = pred_labels.numpy()

    pred_train = np.ravel(pred_labels[:, train_idx])
    pred_test = np.ravel(pred_labels[:, test_idx])
    labels_train = np.ravel(labels[:, train_idx])
    labels_test = np.ravel(labels[:, test_idx])

    result_train = {
        "prec": metrics.precision_score(labels_train, pred_train, average="macro", zero_division=0),
        "recall": metrics.recall_score(labels_train, pred_train, average="macro", zero_division=0),
        "acc": metrics.accuracy_score(labels_train, pred_train),
        "conf_mat": metrics.confusion_matrix(labels_train, pred_train, labels=[0,1]),
    }   if len(labels_train) else {"prec":0.0,"recall":0.0,"acc":0.0,"conf_mat":np.array([[0,0],[0,0]])}
    result_test = {
        "prec": metrics.precision_score(labels_test, pred_test, average="macro", zero_division=0),
        "recall": metrics.recall_score(labels_test, pred_test, average="macro", zero_division=0),
        "acc": metrics.accuracy_score(labels_test, pred_test),
        "conf_mat": metrics.confusion_matrix(labels_test, pred_test, labels=[0,1]),
    }   if len(labels_test) else {"prec":0.0,"recall":0.0,"acc":0.0,"conf_mat":np.array([[0,0],[0,0]])}
    return result_train, result_test

#############################
#
# Run Experiments
#
#############################
def syn_task(args, writer=None):
    if os.path.exists("./workspace/graph.pkl"):
        G, labels, name = torch.load("./workspace/graph.pkl")
    else:
        basis_type, shape_type = args.syn_type.split('-')
        if basis_type=='tree':
            width_basis = 9
        elif basis_type=='ba':
            width_basis = 1023
        if shape_type=='cycle':
            nb_shapes = 120
        elif shape_type=='grid':
            nb_shapes = 80
        elif shape_type=='house':
            nb_shapes = 144
        G, labels, name = gengraph.gen_syn(basis_type, shape_type, width_basis, nb_shapes, 5)
    num_classes = max(labels) + 1
    torch.save([labels[i]>0 for i in range(len(labels))],"./workspace/struct_nodes.pkl")
    
    if args.graph_only:
        plt.figure(figsize=(16, 9), dpi=300)
        nx.draw(G,cmap=plt.get_cmap('jet'),node_color=list(labels),
                with_labels=False,node_size=50,alpha=0.7)
        plt.savefig("./workspace/graph.png"); plt.close()
        exit(0)
    if args.feat_gen in ['Binomial','Correlated','CorrelatedXOR','Independent']:
        if args.feat_gen in ['Binomial']:
            feature_generator = featgen.BinomialFeatureGen(len(labels),args.p,args.seed)
        elif args.feat_gen in ['Correlated','CorrelatedXOR']:
            struct = [j for j in range(len(labels)) if labels[j]>0]
            normal = [j for j in range(len(labels)) if not labels[j]>0]
            feature_generator = featgen.CorrelatedFeatureGen(struct,normal,args.p,args.seed)
        feature_generator.gen_node_features(G)
        torch.save([G.nodes[i]['feat'][0]>0 for i in range(len(labels))],"./workspace/feat_nodes_p={:.2f}.pkl".format(args.p))
        if args.feat_gen in ['Binomial','Correlated','CorrelatedXOR']:
            for i in range(len(labels)):
                labels[i] = (int((labels[i]>0) ^  (G.nodes[i]['feat'][0]>0))
                        if args.feat_gen=='CorrelatedXOR'
                        else int((labels[i]>0) or (G.nodes[i]['feat'][0]>0)))
        torch.save([labels[i]>0 for i in range(len(labels))],"./workspace/labeled_nodes_p={:.2f}.pkl".format(args.p))
    else:
        feature_generator = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        feature_generator.gen_node_features(G)
        torch.save([i for i in range(len(labels))],"./workspace/feat_nodes_p={:.2f}.pkl".format(args.p))
        torch.save([labels[i]>0 for i in range(len(labels))],"./workspace/labeled_nodes_p={:.2f}.pkl".format(args.p))

    ### Insert Code Here


    ###

    if args.method == "attn":
        # print("Method: attn")
        pass
    else:
        # print("Method:", args.method)
        model = models.GcnEncoderNode(
            args.input_dim,
            args.hidden_dim,
            args.output_dim,
            num_classes,
            args.num_gc_layers,
            bn=args.bn,
            args=args,
        )

        if args.gpu:
            model = model.cuda()

    train_node_classifier(G, labels, model, args, writer=writer)

def main():
    prog_args = configs.arg_parse()
    # if prog_args.gpu:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = prog_args.cuda
    #     print("Using CUDA", prog_args.cuda)
    # else:
    #     print("Using CPU")

    # use --bmname=[dataset_name] for Reddit-Binary, Mutagenicity
    syn_task(prog_args, writer=None)


if __name__ == "__main__":
    main()

