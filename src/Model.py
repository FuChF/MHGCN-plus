import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter

from src.Decoupling_matrix_aggregation import adj_matrix_weight_merge
from src.Decoupling_matrix_aggregation import new_adj_matrix_weight_merge

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))#鏉冮噸鐭╅樀
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))#鍋忕Щ鍚戦�?
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        try:
            input = input.float()
        except:
            pass
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    #
    # def __repr__(self):
    #     return self.__class__.__name__ + ' (' \
    #            + str(self.in_features) + ' -> ' \
    #            + str(self.out_features) + ')'

class GCN(nn.Module):
    """
    A Two-layer GCN.
    """
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, use_relu=True):
        x = self.gc1(x, adj)
        if use_relu:
            x = F.relu(x)

        x = self.gc2(x, adj)
        return x

class FAME_GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(FAME_GCN, self).__init__()

        # Multilayer Graph Convolution
        self.gc0 = GraphConvolution(nfeat, out)
        self.gc1 = GraphConvolution(nfeat, out)
        self.gc3 = GraphConvolution(nfeat, out)
        self.gc2 = GraphConvolution(out, out)
        # self.gc3 = GraphConvolution(out, out)
        # self.gc3 = GraphConvolution(out, out)
        # self.gc4 = GraphConvolution(out, out)
        # self.gc5 = GraphConvolution(out, out)
        self.dropout = dropout

        # Alibaba
        # self.weight_b = torch.nn.Parameter(torch.FloatTensor(16, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b,a = 0,b = 0.1)
        # self.weight_b2 = torch.nn.Parameter(torch.FloatTensor(4, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b2, a=0, b=0.1)
        # MHGCN-r
        # self.weight_b = torch.nn.Parameter(torch.FloatTensor(4, 1), requires_grad=False)
        # torch.nn.init.uniform_(self.weight_b, a=1, b=1)

        # DBLP
        # self.weight_b = torch.nn.Parameter(torch.FloatTensor(3, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b, a=0, b=0.1)
        # MHGCN-r
        # self.weight_b = torch.nn.Parameter(torch.FloatTensor(3, 1), requires_grad=False)
        # torch.nn.init.uniform_(self.weight_b, a=1, b=1)

        # self.weight_b = torch.nn.Parameter(torch.zeros(9, 1), requires_grad=True)
        # self.weight_b2 = torch.nn.Parameter(torch.zeros(3, 1), requires_grad=True)
        self.weight_b = torch.nn.Parameter(torch.FloatTensor(9, 1), requires_grad=True)
        torch.nn.init.uniform_(self.weight_b,a = 0,b = 0.1)
        self.weight_b2 = torch.nn.Parameter(torch.FloatTensor(3, 1), requires_grad=True)
        torch.nn.init.uniform_(self.weight_b2, a=0, b=0.1)

        # Aminer
        # self.weight_b = torch.nn.Parameter(torch.FloatTensor(2, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b, a=0, b=0.1)
        # MHGCN-r
        # self.weight_b = torch.nn.Parameter(torch.FloatTensor(2, 1), requires_grad=False)
        # torch.nn.init.uniform_(self.weight_b, a=1, b=1)

        # self.weight_b = torch.nn.Parameter(torch.FloatTensor(9, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b,a = 0,b = 0.1)
        # self.weight_b2 = torch.nn.Parameter(torch.FloatTensor(3, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b2, a=0, b=0.1)
        #链路预测
        # self.weight_b = torch.nn.Parameter(torch.FloatTensor(4, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b,a = 0,b = 0.1)
        # self.weight_b2 = torch.nn.Parameter(torch.FloatTensor(2, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b2, a=0, b=0.1)

        # IMDB
        # self.weight_b = torch.nn.Parameter(torch.FloatTensor(2, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b, a=0, b=0.1)
        # MHGCN-r
        # self.weight_b = torch.nn.Parameter(torch.FloatTensor(2, 1), requires_grad=False)
        # torch.nn.init.uniform_(self.weight_b, a=1, b=1)
        #链路预测
        # self.weight_b2 = torch.nn.Parameter(torch.FloatTensor(3, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b2, a=0, b=0.1)
        # self.weight_b = torch.nn.Parameter(torch.FloatTensor(9, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b,a = 0,b = 0.1)

        # self.weight_b = torch.nn.Parameter(torch.FloatTensor(4, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b,a = 0,b = 0.1)
        # self.weight_b2 = torch.nn.Parameter(torch.FloatTensor(2, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b2, a=0, b=0.1)

        #amazon
        # self.weight_b = torch.nn.Parameter(torch.FloatTensor(4, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b,a = 0,b = 0.1)
        # self.weight_b2 = torch.nn.Parameter(torch.FloatTensor(2, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b2, a=0, b=0.1)

    def forward(self, feature, A,A_t, use_relu=True):
        final_A = adj_matrix_weight_merge(A, self.weight_b2)
        final_A2 = new_adj_matrix_weight_merge(A_t, self.weight_b)
        # final_A0= torch.diag(torch.ones(12772))
        # print(final_A2)
        try:
            feature = torch.tensor(feature.astype(float).toarray())
        except:
            try:
                feature = torch.from_numpy(feature.toarray())
            except:
                pass
        # feature = F.dropout(feature, self.dropout, training=self.training)
        U1 = self.gc3(feature, final_A)

        U2 = self.gc1(feature, final_A2)
        # U2 = self.gc2(U1, final_A)

        # print(self.training)
        # U0 = self.gc0(feature, final_A0)
        # U2= torch.spmm(final_A, U1)
        # U2 = self.gc2(U1, final_A)
        # U3 = self.gc3(U2, final_A)
        # U4 = self.gc4(U2, final_A)
        # U5 = self.gc5(U2, final_A)

        U3=torch.cat((U1, U2), dim=1)
        # U3=(U1+U2)/2

        # return U2
        return U3