import numpy as np
import scipy.io as sio
import pickle as pkl
import torch.nn as nn
from sklearn.metrics import f1_score
from Decoupling_matrix_aggregation import coototensor
import time

import heapq
import torch

from src.logreg import LogReg





def load_data(dataset, datasetfile_type): # Get the label of node classification, training set, verification machine and test set
    if datasetfile_type == 'mat':
        data = sio.loadmat('data/{}.mat'.format(dataset))
    else:
        data = pkl.load(open('data/{}.pkl'.format(dataset), "rb"))
    try:
        labels = data['label']
    except:
        labels = data['labelmat']

    idx_train = data['train_idx'].ravel()
    try:
        idx_val = data['valid_idx'].ravel()
    except:
        idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return labels, idx_train.astype(np.int32) - 1, idx_val.astype(np.int32) - 1, idx_test.astype(np.int32) - 1


def node_classification_evaluate(model, feature, A, file_name, file_type, device, isTest=True):# Training process


    # IMDB
    # p1 = coototensor(A[0][0].tocoo()).to_dense()
    # p1=(p1+p1.transpose(0, 1))
    # p2 = coototensor(A[0][2].tocoo()).to_dense()
    # p2 = (p2 + p2.transpose(0, 1))
    # a = torch.mm(p1,p1)
    # 
    # b = torch.mm(p1,p2)
    # c = torch.mm(p2,p1)
    # d = torch.mm(p2,p2)
    # print("mm complete")
    # A_t = torch.stack([a,b,  c, d], dim=2)



    # Alibaba
    # p1 = coototensor(A[0][0].tocoo()).to_dense()
    # p2 = coototensor(A[1][0].tocoo()).to_dense()
    # p3 = coototensor(A[2][0].tocoo()).to_dense()
    # p4 = coototensor(A[3][0].tocoo()).to_dense()
    # p1 = (p1 + p1.transpose(0, 1))
    # p2 = (p2 + p2.transpose(0, 1))
    # p3 = (p3 + p3.transpose(0, 1))
    # p4 = (p4 + p4.transpose(0, 1))
    # a1 = torch.mm(p1, p1)
    # a2 = torch.mm(p1, p2)
    # a3 = torch.mm(p1, p3)
    # a4 = torch.mm(p1, p4)
    # b1 = torch.mm(p2, p1)
    # b2 = torch.mm(p2, p2)
    # b3 = torch.mm(p2, p3)
    # b4 = torch.mm(p2, p4)
    # c1 = torch.mm(p3, p1)
    # c2 = torch.mm(p3, p2)
    # c3 = torch.mm(p3, p3)
    # c4 = torch.mm(p3, p4)
    # d1 = torch.mm(p4, p1)
    # d2 = torch.mm(p4, p2)
    # d3 = torch.mm(p4, p3)
    # d4 = torch.mm(p4, p4)
    # print("mm complete")
    # A_t = torch.stack([a1,a2,a3,a4,b1,b2,b3,b4,c1,c2,c3,c4,d1,d2,d3,d4], dim=2)


    # Aminer
    # p1 = coototensor(A[0][0].tocoo()).to_dense()
    # p1 = (p1 + p1.transpose(0, 1))
    # p2 = coototensor(A[0][1].tocoo()).to_dense()
    # p2 = (p2 + p2.transpose(0, 1))
    # p3 = coototensor(A[0][2].tocoo()).to_dense()
    # p3 = (p3 + p3.transpose(0, 1))
    # print(torch.sum(torch.sum(p1,dim=0),dim=0))
    # a1 = torch.mm(p1, p1)
    # print(torch.sum(torch.sum(a1, dim=0), dim=0))
    # a2 = torch.mm(p1, p2)
    # print(torch.sum(torch.sum(a2, dim=0), dim=0))
    # print("a complete")
    # a3 = torch.mm(p1, p3)
    # print(torch.sum(torch.sum(a3, dim=0), dim=0))
    # b1 = torch.mm(p2, p1)
    # print(torch.sum(torch.sum(b1, dim=0), dim=0))
    # b2 = torch.mm(p2, p2)
    # print(torch.sum(torch.sum(b2, dim=0), dim=0))
    # b3 = torch.mm(p2, p3)
    # print(torch.sum(torch.sum(b3, dim=0), dim=0))
    # print("b complete")
    # c1 = torch.mm(p3, p1)
    # print(torch.sum(torch.sum(c1, dim=0), dim=0))
    # c2 = torch.mm(p3, p2)
    # print(torch.sum(torch.sum(c2, dim=0), dim=0))
    # c3 = torch.mm(p3, p3)
    # print(torch.sum(torch.sum(c3, dim=0), dim=0))
    # print("c complete")
    # print("mm complete")
    # A_t = torch.stack([a1,a2,a3,b1,b2,b3,c1,c2,c3], dim=2)

    # DBLP
    p1 = coototensor(A[0][0].tocoo()).to_dense()
    p1 = (p1 + p1.transpose(0, 1))
    p2 = coototensor(A[0][1].tocoo()).to_dense()
    p2 = (p2 + p2.transpose(0, 1))
    p3 = coototensor(A[0][2].tocoo()).to_dense()
    p3 = (p3 + p3.transpose(0, 1))
    print(torch.sum(torch.sum(p1,dim=0),dim=0))
    a1 = torch.mm(p1, p1)
    print(torch.sum(torch.sum(a1, dim=0), dim=0))
    a2 = torch.mm(p1, p2)
    print(torch.sum(torch.sum(a2, dim=0), dim=0))
    print("a complete")
    a3 = torch.mm(p1, p3)
    print(torch.sum(torch.sum(a3, dim=0), dim=0))
    b1 = torch.mm(p2, p1)
    print(torch.sum(torch.sum(b1, dim=0), dim=0))
    b2 = torch.mm(p2, p2)
    print(torch.sum(torch.sum(b2, dim=0), dim=0))
    b3 = torch.mm(p2, p3)
    print(torch.sum(torch.sum(b3, dim=0), dim=0))
    print("b complete")
    c1 = torch.mm(p3, p1)
    print(torch.sum(torch.sum(c1, dim=0), dim=0))
    c2 = torch.mm(p3, p2)
    print(torch.sum(torch.sum(c2, dim=0), dim=0))
    c3 = torch.mm(p3, p3)
    print(torch.sum(torch.sum(c3, dim=0), dim=0))
    print("c complete")
    print("mm complete")
    A_t = torch.stack([a1,a2,a3,b1,b2,b3,c1,c2,c3], dim=2)


    embeds = model(feature, A, A_t)

    labels, idx_train, idx_val, idx_test = load_data(file_name, file_type)

    try:
        labels = labels.todense()
    except:
        pass
    labels = labels.astype(np.int16)
    embeds = torch.FloatTensor(embeds[np.newaxis]).to(device)
    labels = torch.FloatTensor(labels[np.newaxis]).to(device)
    idx_train = torch.LongTensor(idx_train).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)

    hid_units = embeds.shape[2]
    nb_classes = labels.shape[2]
    xent = nn.CrossEntropyLoss()
    train_embs = embeds[0, idx_train]
    val_embs = embeds[0, idx_val]
    test_embs = embeds[0, idx_test]
    train_lbls = torch.argmax(labels[0, idx_train], dim=1)
    val_lbls = torch.argmax(labels[0, idx_val], dim=1)
    test_lbls = torch.argmax(labels[0, idx_test], dim=1)

    accs = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = []
    print("start")
    for _ in range(1):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam([{'params': model.parameters(), 'lr': 0.0005}, {'params': log.parameters()}], lr=0.05, weight_decay=0.0)
        log.to(device)

        val_accs = []
        test_accs = []
        val_micro_f1s = []
        test_micro_f1s = []
        val_macro_f1s = []
        test_macro_f1s = []
        print(111111111111111111111111111111111111)
        print(model.parameters())
        print(111111111111111111111111111111111111)
        starttime = time.time()
        for iter_ in range(200):
            # print(111111111111111111111111111111111111)
            # for p in model.named_parameters():
            #     print(p)
            # print(111111111111111111111111111111111111)
            embeds = model(feature, A, A_t)
            # print(embeds)
            embeds = torch.FloatTensor(embeds[np.newaxis]).to(device)
            train_embs = embeds[0, idx_train]
            val_embs = embeds[0, idx_val]
            test_embs = embeds[0, idx_test]

            # train
            log.train()
            print(log.training)
            opt.zero_grad()

            logits = log(train_embs)

            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()


            logits_tra = log(train_embs)
            preds = torch.argmax(logits_tra, dim=1)
            
            tra_f1_macro = f1_score(train_lbls.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='macro')
            tra_f1_micro = f1_score(train_lbls.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='micro')
            print("===============================train{}\t{:.4f}\t{:.4f}\t{:.4f}".format(iter_ + 1, loss.item(), tra_f1_macro,
                                                              tra_f1_micro))

            log.eval()
            print(log.training)
            print('----------------------val-----------------------')
            logits_val = log(val_embs)
            print('------------------------------------------------')
            preds = torch.argmax(logits_val, dim=1)




            val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
            val_f1_macro = f1_score(val_lbls.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='macro')
            val_f1_micro = f1_score(val_lbls.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='micro')

            print("{}\t{:.4f}\t{:.4f}\t{:.4f}".format(iter_ + 1, loss.item(), val_f1_macro,
                                                              val_f1_micro))
            # print("weight_b2:{}".format(model.weight_b2))
            # print("weight_b:{}".format(model.weight_b))
            val_accs.append(val_acc.item())
            # val_macro_f1s.insert(0,val_f1_macro)
            # val_micro_f1s.insert(0,val_f1_micro)
            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)
            # test
            logits_test = log(test_embs)
            preds = torch.argmax(logits_test, dim=1)

            test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            test_f1_macro = f1_score(test_lbls.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='macro')
            test_f1_micro = f1_score(test_lbls.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='micro')
            print("test_f1-ma: {:.4f}\ttest_f1-mi: {:.4f}".format(test_f1_macro, test_f1_micro))

            test_accs.append(test_acc.item())
            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)

        endtime = time.time()



        print('time: {:.10f}'.format(endtime - starttime))
        max_iter = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter])
        # max_iter = 149-val_macro_f1s.index(max(val_macro_f1s))
        max_iter = val_macro_f1s.index(max(val_macro_f1s))
        macro_f1s.append(test_macro_f1s[max_iter])
        print(max_iter)

        # max_iter = 149-val_micro_f1s.index(max(val_micro_f1s))
        max_iter = val_micro_f1s.index(max(val_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_iter])
        print(max_iter)



    if isTest:
        print("\t[Classification] Macro-F1: {:.4f} ({:.4f}) | Micro-F1: {:.4f} ({:.4f})".format(np.mean(macro_f1s),
                                                                                                np.std(macro_f1s),
                                                                                                np.mean(micro_f1s),
                                                                                                np.std(micro_f1s)))
    else:
        return np.mean(macro_f1s), np.mean(micro_f1s)

    return np.mean(macro_f1s), np.mean(micro_f1s)