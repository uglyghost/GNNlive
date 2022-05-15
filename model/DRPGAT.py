import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd  import Variable

import numpy as np


class DRPGAT(nn.Module):
    """
    act: activation function for GAT
    n_node: number of nodes on the network
    output_dim: output embed size for GAT
    seq_len: number of graphs
    n_heads: number of heads for GAT
    attn_drop: attention/coefficient matrix dropout rate
    ffd_drop: feature matrix dropout rate
    residual: if using short cut or not for GRU network
    """

    def __init__(self,
                 n_node,
                 input_dim,
                 output_dim,
                 seq_len,
                 n_heads,
                 attn_drop,
                 ffd_drop,
                 residual=False,
                 bias=True,
                 sparse_inputs=False
                 ):
        super(DRPGAT, self).__init__()

        self.act = nn.ELU()
        self.n_node = n_node
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.n_heads = n_heads
        self.attn_drop = attn_drop
        self.ffd_drop = ffd_drop
        self.residual = residual

        self.var = {}

        self.evolve_weights = GRU(n_node, input_dim, output_dim, n_heads, residual)
        self.model = GAT(self.input_dim, self.output_dim, self.n_heads, self.attn_drop,
                         self.ffd_drop)

    def call(self, adjs, feats, p_covss):
        embeds = []
        adj = adjs[0]
        feat = feats[0]
        weight_vars = {}

        for i in range(self.n_heads):
            weight_var = torch.randn(self.output_dim, self.input_dim)

            weight_vars[i] = weight_var

            self.var['weight_var_' + str(i)] = weight_var

        output = self.model(feat, adj, p_covss[0])
        #         print(output.shape)
        embed = output.view([-1, self.output_dim])
        embeds.append(embed)

        for i in range(1, self.seq_len):
            adj = adjs[i]
            feat = feats[i]

            weight_vars = self.evolve_weights(adj, weight_vars)

            output = self.model(feat, adj, p_covss[i])
            embed = output.view([-1, self.output_dim])
            embeds.append(embed)

        return embeds

    def pred(self, feats_p_u, feats_p_t, feats_n_u, feats_n_t):
        p_score = torch.mm(feats_p_u, feats_p_t.T, out=None).reshape(-1)
        n_score = torch.mm(feats_n_u, feats_n_t.T, out=None).reshape(-1)

        return p_score, n_score

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)


class GRU(nn.Module):
    def __init__(self, n_node, input_dim, output_dim, n_head, residual=False):
        super(GRU, self).__init__()
        self.n_node = n_node
        self.n_head = n_head
        self.residual = residual
        self.gru_cell = GRU_cell(self.n_node, input_dim, output_dim)

    def call(self, adj_mat, weight_vars):
        weight_vars_next = {}
        adj_mat = spy_sparse2torch_sparse(adj_mat)
        for i in range(self.n_head):
            if self.residual:
                new_Q = self.gru_cell(adj_mat, weight_vars[i]) + weight_vars[i]
            else:
                new_Q = self.gru_cell(adj_mat, weight_vars[i])
            weight_vars_next[i] = new_Q
        return weight_vars_next

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)


class GRU_cell(nn.Module):
    def __init__(self, n_node, input_dim, output_dim):
        super(GRU_cell, self).__init__()
        self.n_node = n_node

        self.outlayer1 = nn.Sigmoid()
        self.outlayer2 = nn.Sigmoid()
        self.outlayer3 = nn.Tanh()

        self.reset = GRU_gate(n_node, input_dim, output_dim, self.outlayer1)
        self.update = GRU_gate(n_node, input_dim, output_dim, self.outlayer2)
        self.htilda = GRU_gate(n_node, input_dim, output_dim, self.outlayer3)

    def call(self, adj_mat, prev_w):
        reset = self.reset(adj_mat, prev_w)
        update = self.update(adj_mat, prev_w)

        h_cap = reset * prev_w
        h_cap = self.htilda(adj_mat, h_cap)
        new_Q = (1 - update) * prev_w + update * h_cap

        return new_Q

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)


class GRU_gate(nn.Module):
    def __init__(self, n_node, input_dim, output_dim, act, reduce=False):
        super(GRU_gate, self).__init__()
        self.activation = act
        self.reduce = reduce

        self.W = glorot([n_node, output_dim])
        self.U = glorot([output_dim, output_dim])
        self.bias = zeros([input_dim, output_dim])
        if n_node != input_dim:
            self.reduce = True
            self.P = glorot([input_dim, n_node])

    def call(self, adj_mat, prev_w):
        #         out = self.activation(self.W.matmul(x) + \
        #                               self.U.matmul(hidden) + \
        #                               self.bias)
        if self.reduce:
            temp_ = dot_mat(x=adj_mat, y=self.W, sparse=True)
            out = self.activation(dot_mat(x=self.P, y=temp_) +
                                  dot_mat(x=prev_w, y=self.U) +
                                  self.bias)
        else:
            out = self.activation(dot_mat(x=adj_mat, y=self.W, sparse=True) +
                                  dot_mat(x=prev_w, y=self.U) +
                                  self.bias)
        return out

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)


class Attn_head(nn.Module):
    def __init__(self,
                 in_channel,
                 out_sz,
                 in_drop=0.0,
                 coef_drop=0.0,
                 activation=None,
                 residual=False):
        super(Attn_head, self).__init__()
        self.in_channel = in_channel
        self.out_sz = out_sz
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.activation = activation
        self.residual = residual

        self.conv1 = nn.Conv1d(self.in_channel, self.out_sz, 1)
        self.conv2_1 = nn.Conv1d(self.out_sz, 1, 1)
        self.conv2_2 = nn.Conv1d(self.out_sz, 1, 1)
        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        # pytorch中dropout的参数p表示每个神经元一定概率失活
        self.in_dropout = nn.Dropout()
        self.coef_dropout = nn.Dropout()
        self.res_conv = nn.Conv1d(self.in_channel, self.out_sz, 1)

    def forward(self, x, adj, p_covs):
        seq = torch.tensor(x, dtype=torch.float32)
        if self.in_drop != 0.0:
            seq = self.in_dropout(x)
        seq_fts = self.conv1(seq.reshape([55, self.in_channel, 1]))
        f_1 = self.conv2_1(seq_fts)
        f_2 = self.conv2_2(seq_fts)
        logits = f_1 + torch.transpose(f_2, 2, 1)
        logits = self.leakyrelu(logits)
        coefs = self.softmax(logits)
        if self.coef_drop != 0.0:
            coefs = self.coef_dropout(coefs)
        if self.in_dropout != 0.0:
            seq_fts = self.in_dropout(seq_fts)
        ret = torch.matmul(coefs, torch.transpose(seq_fts, 2, 1))
        ret = torch.transpose(ret, 2, 1)
        if self.residual:
            if seq.shape[1] != ret.shape[1]:
                ret = ret + self.res_conv(seq)
            else:
                ret = ret + seq
        return self.activation(ret)


class GAT(nn.Module):
    def __init__(self,
                 nb_nodes,
                 nb_classes,
                 n_heads,
                 attn_drop,
                 ffd_drop,
                 hid_units=200,
                 residual=False):
        super(GAT, self).__init__()
        self.nb_classes = nb_classes
        self.nb_nodes = nb_nodes
        self.attn_drop = attn_drop
        self.ffd_drop = ffd_drop
        self.hid_units = hid_units
        self.n_heads = n_heads
        self.residual = residual
        self.act = nn.ELU()

        self.attn1 = Attn_head(in_channel=200, out_sz=self.hid_units,
                               in_drop=self.ffd_drop,
                               coef_drop=self.attn_drop, activation=self.act,
                               residual=self.residual)
        self.attn2 = Attn_head(in_channel=1600, out_sz=self.nb_classes,
                               in_drop=self.ffd_drop,
                               coef_drop=self.attn_drop, activation=self.act,
                               residual=self.residual)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, adj, p_covs):
        attns = []
        for _ in range(self.n_heads):
            attns.append(self.attn1(x, adj, p_covs))
        h_1 = torch.cat(attns, dim=1)
        out = self.attn2(h_1, adj, p_covs)
        logits = torch.transpose(out.view(self.nb_classes, -1), 1, 0)
        logits = self.softmax(logits)
        return logits


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = 2 * init_range * torch.rand(shape, requires_grad=True, dtype=torch.float32) - init_range
    return torch.Tensor(initial)


def zeros(shape, name=None):
    """All zeros."""
    initial = torch.zeros(shape, dtype=torch.float32)
    return torch.Tensor(initial)


def dot_mat(x, y, sparse=False):
    if sparse:
        return torch.sparse.mm(x, y)
    return torch.matmul(x, y)

def spy_sparse2torch_sparse(data):
    samples=data.shape[0]
    features=data.shape[1]
    values=data.data
    coo_data=data.tocoo()
    indices=torch.LongTensor([coo_data.row,coo_data.col])
    t=torch.sparse.FloatTensor(indices,torch.from_numpy(values).float(),[samples,features])
    return t