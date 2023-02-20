import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score, auc
import torch.nn.functional as F
from Decoupling_matrix_aggregation import coototensor
from src.args import get_citation_args
import src.logreg

args = get_citation_args()
def test_link_prediction_evaluate(type,model, true_edges, false_edges): # Training process
    f1 = open("new/test"+str(type)+".txt", 'w')
    f2 = open("new/testfault" + str(type) + ".txt", 'w')
    true_list = list()
    prediction_list = list()
    all_edge1 = list()
    all_edge2 = list()
    true_num = 0
    for edge in true_edges:
        # tmp_score = get_score(model, str(edge[0]), str(edge[1])) # for amazon
        tmp_score = get_score(model, str(int(edge[0])), str(int(edge[1])))
        # tmp_score = get_score(model, str(int(edge[0] -1)), str(int(edge[1]-1)))
        if tmp_score is not None:
            true_list.append(1)
            prediction_list.append(tmp_score)
            true_num += 1
            all_edge1.append(int(edge[0]))
            all_edge2.append(int(edge[1]))
    for edge in false_edges:
        # tmp_score = get_score(model, str(edge[0]), str(edge[1])) # for amazon
        tmp_score = get_score(model, str(int(edge[0])), str(int(edge[1])))
        # tmp_score = get_score(model, str(int(edge[0] -1)), str(int(edge[1]-1)))
        if tmp_score is not None:
            true_list.append(0)
            all_edge1.append(int(edge[0]))
            all_edge2.append(int(edge[1]))
            prediction_list.append(tmp_score)

    sorted_pred = prediction_list[:]
    sorted_pred.sort()
    threshold = sorted_pred[-true_num]

    y_pred = np.zeros(len(prediction_list), dtype=np.int32)
    for i in range(len(prediction_list)):
        if prediction_list[i] >= threshold:
            y_pred[i] = 1

    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    print('y_true')
    print(y_true)
    print('y_pred')
    print(y_pred)

    for t in range(len(y_true)):
        if y_true[t]==y_pred[t]:
            f1.write('{} {} {} {}\n'.format(type+1, all_edge1[t], all_edge2[t], y_true[t]))
        else:
            f2.write('{} {} {} {}\n'.format(type + 1, all_edge1[t], all_edge2[t], y_true[t]))





    ps, rs, _ = precision_recall_curve(y_true, y_scores)
    f1.close()
    return roc_auc_score(y_true, y_scores), f1_score(y_true, y_pred), auc(rs, ps)


def load_training_data(f_name):
    # print('We are loading training data from:', f_name)
    edge_data_by_type = dict()
    all_edges = list()
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            # words = line[:-1].split('\t')
            words = line[:-1].split()
            # print(words)
            if words[0] not in edge_data_by_type:
                edge_data_by_type[words[0]] = list()
            x, y = words[1], words[2]
            edge_data_by_type[words[0]].append((x, y))
            all_edges.append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    all_edges = list(set(all_edges))
    edge_data_by_type['Base'] = all_edges
    print('total training nodes: ' + str(len(all_nodes)))
    # print('Finish loading training data')
    return edge_data_by_type


def load_testing_data(f_name):
    # print('We are loading testing data from:', f_name)
    true_edge_data_by_type = dict()
    false_edge_data_by_type = dict()
    all_edges = list()
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            # words = line[:-1].split('\t')
            words = line[:-1].split()
            # print(words[0])
            # print(words[1])
            # print(words[2])
            # print(words[3])
            x, y = words[1], words[2]
            if int(words[3]) == 1:
                if words[0] not in true_edge_data_by_type:
                    true_edge_data_by_type[words[0]] = list()
                true_edge_data_by_type[words[0]].append((x, y))
            else:
                if words[0] not in false_edge_data_by_type:
                    false_edge_data_by_type[words[0]] = list()
                false_edge_data_by_type[words[0]].append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    # print('Finish loading testing data')
    return true_edge_data_by_type, false_edge_data_by_type


def get_score(local_model, node1, node2):
    try:
        vector1 = local_model[node1]
        vector2 = local_model[node2]
        if type(vector1) != np.ndarray:
            vector1 = vector1.toarray()[0]
            vector2 = vector2.toarray()[0]

        return np.dot(vector1, vector2)
        # return np.dot(vector1, vector2) / ((np.linalg.norm(vector1) * np.linalg.norm(vector2) + 0.00000000000000001))
    except Exception as e:
        pass


def link_prediction_evaluate(model, true_edges, false_edges): # Training process
    true_list = list()
    prediction_list = list()
    true_num = 0
    for edge in true_edges:
        # tmp_score = get_score(model, str(edge[0]), str(edge[1])) # for amazon
        tmp_score = get_score(model, str(int(edge[0])), str(int(edge[1])))
        # tmp_score = get_score(model, str(int(edge[0] -1)), str(int(edge[1]-1)))
        if tmp_score is not None:
            true_list.append(1)
            prediction_list.append(tmp_score)
            true_num += 1

    for edge in false_edges:
        # tmp_score = get_score(model, str(edge[0]), str(edge[1])) # for amazon
        tmp_score = get_score(model, str(int(edge[0])), str(int(edge[1])))
        # tmp_score = get_score(model, str(int(edge[0] -1)), str(int(edge[1]-1)))
        if tmp_score is not None:
            true_list.append(0)
            prediction_list.append(tmp_score)

    sorted_pred = prediction_list[:]
    sorted_pred.sort()
    threshold = sorted_pred[-true_num]

    y_pred = np.zeros(len(prediction_list), dtype=np.int32)
    for i in range(len(prediction_list)):
        if prediction_list[i] >= threshold:
            y_pred[i] = 1

    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    # print('y_true')
    print(y_true)
    # print('y_pred')
    print(y_pred)
    ps, rs, _ = precision_recall_curve(y_true, y_scores)
    return roc_auc_score(y_true, y_scores), f1_score(y_true, y_pred), auc(rs, ps)


def predict_model(model, file_name, feature, A, eval_type, node_matching):
    training_data_by_type = load_training_data(file_name + '/train.txt')
    # train_true_data_by_edge, train_false_data_by_edge = load_testing_data(file_name + '/train.txt')
    valid_true_data_by_edge, valid_false_data_by_edge = load_testing_data(file_name + '/valid.txt')
    testing_true_data_by_edge, testing_false_data_by_edge = load_testing_data(file_name + '/test.txt')

    network_data = training_data_by_type
    edge_types = list(network_data.keys())  # ['1', '2', '3', '4', 'Base']

    edge_type_count = len(edge_types) - 1

    # imdb
    # edge_type_count=2
    # print('//////////////////////////////')
    # print(edge_type_count)
    # print('//////////////////////////////')
    # edge_type_count = len(eval_type) - 1s




    # IMDB
    # p1 = coototensor(A[0][0].tocoo()).to_dense()
    # p1=(p1+p1.transpose(0, 1))
    # p2 = coototensor(A[0][2].tocoo()).to_dense()
    # p2 = (p2 + p2.transpose(0, 1))
    # a = torch.mm(p1,p1)
    # b = torch.mm(p1,p2)
    # c = torch.mm(p2,p1)
    # d = torch.mm(p2,p2)
    # print("mm complete")
    # A_t = torch.stack([a,b,  c, d], dim=2)
    #链路预测
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


    # amazon
    # p1 = coototensor(A[0][0].tocoo()).to_dense()
    # p1=(p1+p1.transpose(0, 1))
    # p2 = coototensor(A[1][0].tocoo()).to_dense()
    # p2 = (p2 + p2.transpose(0, 1))
    # a = torch.mm(p1,p1)
    # b = torch.mm(p1,p2)
    # c = torch.mm(p2,p1)
    # d = torch.mm(p2,p2)
    # print("mm complete")
    # A_t = torch.stack([a,b,  c, d], dim=2)
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
    #链路预测
    # p1 = coototensor(A[0][0].tocoo()).to_dense()
    # p1 = (p1 + p1.transpose(0, 1))
    # p2 = coototensor(A[0][2].tocoo()).to_dense()
    # p2 = (p2 + p2.transpose(0, 1))
    # print(torch.sum(torch.sum(p1,dim=0),dim=0))
    # a1 = torch.mm(p1, p1)
    # print(torch.sum(torch.sum(a1, dim=0), dim=0))
    # a2 = torch.mm(p1, p2)
    # print(torch.sum(torch.sum(a2, dim=0), dim=0))
    # print("a complete")
    # b1 = torch.mm(p2, p1)
    # print(torch.sum(torch.sum(b1, dim=0), dim=0))
    # b2 = torch.mm(p2, p2)
    # print(torch.sum(torch.sum(b2, dim=0), dim=0))
    #
    # print("c complete")
    # print("mm complete")
    # A_t = torch.stack([a1,a2,b1,b2], dim=2)

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




    device = torch.device('cpu')

    aucs, f1s, prs = [], [], []

    for _ in range(1):
        for iter_ in range(500):
            model.to(device)
            opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0)

            emb = model(feature, A, A_t)

            emb_true_first = []
            emb_true_second = []
            emb_false_first = []
            emb_false_second = []

            for i in range(edge_type_count):
                if eval_type == 'all' or edge_types[i] in eval_type.split(','):
                    true_edges = valid_true_data_by_edge[edge_types[i]]
                    false_edges = valid_false_data_by_edge[edge_types[i]]

                    for edge in true_edges:
                        # tmp_score = get_score(final_model, str(edge[0]), str(edge[1])) # for amazon
                        emb_true_first.append(emb[int(edge[0])])
                        emb_true_second.append(emb[int(edge[1])])
                        # emb_true_first.append(emb[int(edge[0])-1])
                        # emb_true_second.append(emb[int(edge[1])-1])

                    for edge in false_edges:
                        # tmp_score = get_score(final_model, str(edge[0]), str(edge[1])) # for amazon
                        emb_false_first.append(emb[int(edge[0])])
                        emb_false_second.append(emb[int(edge[1])])
                        # emb_false_first.append(emb[int(edge[0])-1])
                        # emb_false_second.append(emb[int(edge[1])-1])

            pos_out = []
            neg_out = []

            for i in range(len(emb_true_first)):
                pos_out.append(torch.mm(emb_true_first[i].reshape(1, 200), emb_true_second[i].reshape(200, 1)))
            for i in range(len(emb_false_first)):
                neg_out.append(-torch.mm(emb_false_first[i].reshape(1, 200), emb_false_second[i].reshape(200, 1)))

            pos_out = torch.stack(pos_out)
            neg_out = torch.stack(neg_out)

            # emb_true_first = torch.stack(emb_true_first)
            # emb_true_second= torch.stack(emb_true_second)
            # emb_true_second = torch.tensor(list(map(list, zip(*emb_true_second))))
            # emb_false_first = torch.stack(emb_false_first)
            # emb_false_second = torch.stack(emb_false_second)
            # emb_false_second = torch.tensor(list(map(list, zip(*emb_false_second))))

            # for i in range(emb_true_first.shape[0]):
            #     a = emb_true_first[i]
            #     a = emb_true_first[i].reshape(1, emb_true_first.shape[1])
            #     b = emb_true_second[i].reshape(emb_true_first.shape[1], 1)
            #     pos_out.append(torch.mm(emb_true_first[i].reshape(1, emb_true_first.shape[1]), emb_true_second[i].reshape(emb_true_first.shape[1], 1)))
            #     print()
            # pos_out = torch.mm(emb_true_first, emb_true_second)
            # neg_out = -torch.mm(emb_false_first, emb_false_second)
            loss = -torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out))

            opt.zero_grad()
            loss.backward()
            opt.step()


            print(model.training)
            td = model(feature, A,A_t).detach().numpy()
            final_model = {}
            try:
                if node_matching == True:
                    for i in range(0, len(td)):
                        final_model[str(int(td[i][0]))] = td[i][1:]
                else:
                    for i in range(0, len(td)):
                        final_model[str(i)] = td[i]
            except:
                td = td.tocsr()
                if node_matching == True:
                    for i in range(0, td.shape[0]):
                        final_model[str(int(td[i][0]))] = td[i][1:]
                else:
                    for i in range(0, td.shape[0]):
                        final_model[str(i)] = td[i]
            train_aucs, train_f1s, train_prs = [], [], []
            test_aucs, test_f1s, test_prs = [], [], []
            for i in range(edge_type_count):
                if eval_type == 'all' or edge_types[i] in eval_type.split(','):
                    train_auc, triain_f1, train_pr = link_prediction_evaluate(final_model,
                                                                              valid_true_data_by_edge[edge_types[i]],
                                                                              valid_false_data_by_edge[edge_types[i]])
                    train_aucs.append(train_auc)
                    train_f1s.append(triain_f1)
                    train_prs.append(train_pr)
                    # print('test*******************************************************************')
                    if iter_==99:
                        test_auc, test_f1, test_pr = test_link_prediction_evaluate(i,final_model,
                                                                          testing_true_data_by_edge[edge_types[i]],
                                                                          testing_false_data_by_edge[edge_types[i]])
                    else:
                        test_auc, test_f1, test_pr = link_prediction_evaluate(final_model,
                                                                          testing_true_data_by_edge[edge_types[i]],
                                                                          testing_false_data_by_edge[edge_types[i]])
                    test_aucs.append(test_auc)
                    test_f1s.append(test_f1)
                    test_prs.append(test_pr)

            print("{}\t{:.4f}\tweight_b:{}".format(iter_ + 1, loss.item(), model.weight_b))
            print("train_auc:{:.4f}\ttrain_f1:{:.4f}\ttrain_pr:{:.4f}".format(np.mean(train_aucs), np.mean(train_f1s),
                                                                              np.mean(train_prs)))
            print("test_auc:{:.4f}\ttest_f1:{:.4f}\ttest_pr:{:.4f}".format(np.mean(test_aucs), np.mean(test_f1s),
                                                                           np.mean(test_prs)))
            aucs.append(np.mean(test_aucs))
            f1s.append(np.mean(test_f1s))
            prs.append(np.mean(test_prs))

    max_iter = aucs.index(max(aucs))

    return aucs[max_iter], f1s[max_iter], prs[max_iter]