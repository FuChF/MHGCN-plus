import numpy as np
import torch
from scipy.sparse import coo_matrix


def coototensor(A):
    values = A.data
    indices = np.vstack((A.row, A.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = A.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def adj_matrix_weight_merge(A, adj_weight): # Multiplex Relation Aggregation
    # print(adj_weight)
    N = A[0][0].shape[0]
    temp = coo_matrix((N, N))
    temp = coototensor(temp)


    # Alibaba
    # a = coototensor(A[0][0].tocoo())
    # a=a+a.transpose(0, 1)
    # b = coototensor(A[1][0].tocoo())
    # b=b+b.transpose(0, 1)
    # c = coototensor(A[2][0].tocoo())
    # c = c + c.transpose(0, 1)
    # d = coototensor(A[3][0].tocoo())
    # d = d + d.transpose(0, 1)
    # A_t = torch.stack([a, b, c, d], dim=2).to_dense()

    # DBLP
    a = coototensor(A[0][0].tocoo()).to_dense()
    a=a+a.transpose(0, 1)
    b = coototensor(A[0][1].tocoo()).to_dense()
    b=b+b.transpose(0, 1)
    c = coototensor(A[0][2].tocoo()).to_dense()
    c=c+c.transpose(0, 1)
    A_t = torch.stack([a, b, c], dim=2)

    # Aminer
    # a = coototensor(A[0][0].tocoo())
    # a=a+a.transpose(0, 1)
    # b = coototensor(A[0][1].tocoo())
    # b = b + b.transpose(0, 1)
    # c = coototensor(A[0][2].tocoo())
    # c = c + c.transpose(0, 1)
    # A_t = torch.stack([a,b, c], dim=2).to_dense()

    # 链路预测
    # a = coototensor(A[0][0].tocoo())
    # a=a+a.transpose(0, 1)
    # c = coototensor(A[0][2].tocoo())
    # c = c + c.transpose(0, 1)
    # A_t = torch.stack([a, c], dim=2).to_dense()

    # amazon
    # a = coototensor(A[0][0].tocoo()).to_dense()
    # a=a+a.transpose(0, 1)
    # b = coototensor(A[1][0].tocoo()).to_dense()
    # b=b+b.transpose(0, 1)
    # A_t = torch.stack([a, b], dim=2)

    # IMDB
    # a = coototensor(A[0][0].tocoo()).to_dense()
    # a=a+a.transpose(0, 1)
    # b = coototensor(A[0][2].tocoo()).to_dense()
    # b=b+b.transpose(0, 1)
    # A_t = torch.stack([a, b], dim=2)
    # 链路预测
    # a = coototensor(A[0][0].tocoo())
    # a=a+a.transpose(0, 1)
    # b = coototensor(A[0][1].tocoo())
    # b = b + b.transpose(0, 1)
    # c = coototensor(A[0][2].tocoo())
    # c = c + c.transpose(0, 1)
    # A_t = torch.stack([a,b, c], dim=2).to_dense()
    #
    temp = torch.matmul(A_t, adj_weight)
    temp = torch.squeeze(temp, 2)
    # temp = temp.to_sparse()

    return temp

def new_adj_matrix_weight_merge(A_t, adj_weight):




    temp = torch.matmul(A_t, adj_weight)
    temp = torch.squeeze(temp)
    # temp = temp.to_sparse()

    return temp
    # return temp + temp.transpose(0, 1)