import torch
import torch.nn as nn
from torch.nn import init
from dgl.nn.pytorch import GraphConv, HeteroGraphConv
from torch.nn.parameter import Parameter
import dgl.function as fn
import torch.nn.functional as F


class DotProductPredictor(nn.Module):
    def forward(self, graph, h):
        # h是从5.1节的GNN模型中计算出的节点表示
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return graph.edata['score']


class MatGRUCell(torch.nn.Module):
    """
    GRU cell for matrix, similar to the official code.
    Please refer to section 3.4 of the paper for the formula.
    """

    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.update = MatGRUGate(in_feats,
                                 out_feats,
                                 torch.nn.Sigmoid())

        self.reset = MatGRUGate(in_feats,
                                out_feats,
                                torch.nn.Sigmoid())

        self.htilda = MatGRUGate(in_feats,
                                 out_feats,
                                 torch.nn.Tanh())

    def forward(self, prev_Q, z_topk=None):
        if z_topk is None:
            z_topk = prev_Q

        update = self.update(z_topk, prev_Q)
        reset = self.reset(z_topk, prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q


class MatGRUGate(torch.nn.Module):
    """
    GRU gate for matrix, similar to the official code.
    Please refer to section 3.4 of the paper for the formula.
    """

    def __init__(self, rows, cols, activation):
        super().__init__()
        self.activation = activation
        self.W = Parameter(torch.Tensor(rows, rows))
        self.U = Parameter(torch.Tensor(rows, rows))
        self.bias = Parameter(torch.Tensor(rows, cols))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.W)
        init.xavier_uniform_(self.U)
        init.zeros_(self.bias)

    def forward(self, x, hidden):
        out = self.activation(self.W.matmul(x) + \
                              self.U.matmul(hidden) + \
                              self.bias)

        return out


class TopK(torch.nn.Module):
    """
    Similar to the official `egcn_h.py`. We only consider the node in a timestamp based subgraph,
    so we need to pay attention to `K` should be less than the min node numbers in all subgraph.
    Please refer to section 3.4 of the paper for the formula.
    """

    def __init__(self, feats, k):
        super().__init__()
        self.scorer = Parameter(torch.Tensor(feats, 1))
        self.reset_parameters()

        self.k = k

    def reset_parameters(self):
        init.xavier_uniform_(self.scorer)

    def forward(self, node_embs):
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()
        vals, topk_indices = scores.view(-1).topk(self.k)
        out = node_embs[topk_indices] * torch.tanh(scores[topk_indices].view(-1, 1))
        # we need to transpose the output
        return out.t()


class EvolveGCNH(nn.Module):
    def __init__(self, in_feats=166, n_hidden=76, num_layers=2, n_classes=2, classifier_hidden=510):
        # default parameters follow the official config
        super(EvolveGCNH, self).__init__()
        self.num_layers = num_layers
        self.pooling_layers = nn.ModuleList()
        self.recurrent_layers = nn.ModuleList()
        self.gnn_convs = nn.ModuleList()
        self.gcn_weights_list = nn.ParameterList()

        self.pooling_layers.append(TopK(in_feats, n_hidden))
        # similar to EvolveGCNO
        self.recurrent_layers.append(MatGRUCell(in_feats=in_feats, out_feats=n_hidden))
        self.gcn_weights_list.append(Parameter(torch.Tensor(in_feats, n_hidden)))
        self.gnn_convs.append(
            GraphConv(in_feats=in_feats, out_feats=n_hidden, bias=False, activation=nn.RReLU(), weight=False))
        for _ in range(num_layers - 1):
            self.pooling_layers.append(TopK(n_hidden, n_hidden))
            self.recurrent_layers.append(MatGRUCell(in_feats=n_hidden, out_feats=n_hidden))
            self.gcn_weights_list.append(Parameter(torch.Tensor(n_hidden, n_hidden)))
            self.gnn_convs.append(
                GraphConv(in_feats=n_hidden, out_feats=n_hidden, bias=False, activation=nn.RReLU(), weight=False))

        self.mlp = nn.Sequential(nn.Linear(n_hidden, classifier_hidden),
                                 nn.ReLU(),
                                 nn.Linear(classifier_hidden, n_classes))
        self.reset_parameters()

    def reset_parameters(self):
        for gcn_weight in self.gcn_weights_list:
            init.xavier_uniform_(gcn_weight)

    def forward(self, g_list):
        feature_list = []
        for g in g_list:
            feature_list.append(g.ndata['feat'])
        for i in range(self.num_layers):
            W = self.gcn_weights_list[i]
            for j, g in enumerate(g_list):
                X_tilde = self.pooling_layers[i](feature_list[j])
                W = self.recurrent_layers[i](W, X_tilde)
                feature_list[j] = self.gnn_convs[i](g, feature_list[j], weight=W)
        return self.mlp(feature_list[-1])


class EvolveGCNO(nn.Module):
    def __init__(self, in_feats=166, n_hidden=200, num_layers=2, n_classes=100, classifier_hidden=307):
        # default parameters follow the official config
        super(EvolveGCNO, self).__init__()
        self.num_layers = num_layers
        self.recurrent_layers = nn.ModuleList()
        self.gnn_convs = nn.ModuleList()
        self.gcn_weights_list = nn.ParameterList()

        # In the paper, EvolveGCN-O use LSTM as RNN layer. According to the official code,
        # EvolveGCN-O use GRU as RNN layer. Here we follow the official code.
        # See: https://github.com/IBM/EvolveGCN/blob/90869062bbc98d56935e3d92e1d9b1b4c25be593/egcn_o.py#L53
        # PS: I try to use torch.nn.LSTM directly,
        #     like [pyg_temporal](github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/torch_geometric_temporal/nn/recurrent/evolvegcno.py)
        #     but the performance is worse than use torch.nn.GRU.
        # PPS: I think torch.nn.GRU can't match the manually implemented GRU cell in the official code,
        #      we follow the official code here.
        self.recurrent_layers.append(MatGRUCell(in_feats=in_feats, out_feats=n_hidden))
        # Attention: Some people think that the weight of GCN should not be trained, which may require attention.
        # see: https://github.com/benedekrozemberczki/pytorch_geometric_temporal/issues/80#issuecomment-910193561
        self.gcn_weights_list.append(Parameter(torch.Tensor(in_feats, n_hidden)))

        self.gnn_convs.append(
            GraphConv(in_feats=in_feats, out_feats=n_hidden, bias=True, activation=nn.RReLU(), weight=False))
        for _ in range(num_layers - 1):
            self.recurrent_layers.append(MatGRUCell(in_feats=n_hidden, out_feats=n_hidden))
            self.gcn_weights_list.append(Parameter(torch.Tensor(n_hidden, n_hidden)))
            self.gnn_convs.append(
                GraphConv(in_feats=n_hidden, out_feats=n_hidden, bias=True, activation=nn.RReLU(), weight=False))

        self.mlp = nn.Sequential(nn.Linear(n_hidden, classifier_hidden),
                                 nn.ReLU(),
                                 nn.Linear(classifier_hidden, n_classes))
        self.reset_parameters()
        self.pred = DotProductPredictor()

    def reset_parameters(self):
        for gcn_weight in self.gcn_weights_list:
            init.xavier_uniform_(gcn_weight)

    def forward(self, g_list, ng_list, node_feature):
        feature_list_p = []
        # feature_list_n = []
        for i, g in enumerate(g_list):
            feature_list_p.append(node_feature[i])
            # feature_list_n.append(g.ndata['feat'])

        for i in range(self.num_layers):
            W = self.gcn_weights_list[i]
            # W_n = self.gcn_weights_list[i]
            for j, g in enumerate(g_list):
                W = self.recurrent_layers[i](W)
                feature_list_p[j] = self.gnn_convs[i](g, feature_list_p[j], weight=W)

            # for k, ng in enumerate(ng_list):
                # W_n = self.recurrent_layers[i](W_n)
                # feature_list_n[j] = self.gnn_convs[i](ng_list[j], feature_list_n[j], weight=W)

        pos_score = []
        neg_score = []
        for i in range(len(g_list)):
            feature_list_p[i] = self.mlp(feature_list_p[i])
            if i == 0:
                pos_score = self.pred(g_list[i], feature_list_p[i])
                neg_score = self.pred(ng_list[i], feature_list_p[i])
            else:
                pos_score = torch.cat((pos_score, self.pred(g_list[i], feature_list_p[i])), 0)
                neg_score = torch.cat((neg_score, self.pred(ng_list[i], feature_list_p[i])), 0)
        '''

        pos_score = self.pred(g_list[-1], feature_list)
        neg_score = self.pred(ng_list[-1], feature_list)
        '''


        # pos_score = 100*F.normalize(pos_score, p=2, dim=0)
        # neg_score = 100*F.normalize(neg_score, p=2, dim=0)
        # return self.mlp(feature_list[-1])
        # pos_score = pos_score
        # neg_score = neg_score
        return pos_score, neg_score

    def saga(self, g_list, inputs):
        feature_list_p = []
        for i, g in enumerate(g_list):
            feature_list_p.append(inputs[i])

        for i in range(self.num_layers):
            W = self.gcn_weights_list[i]
            # W_n = self.gcn_weights_list[i]
            for j, g in enumerate(g_list):
                W = self.recurrent_layers[i](W)
                feature_list_p[j] = self.gnn_convs[i](g, feature_list_p[j], weight=W)

        feature_list_p = self.mlp(feature_list_p[-1])

        return feature_list_p

    def predict(self, user_emb, tile_emb, thred):
        adj_rec = torch.mm(user_emb, tile_emb.T)
        thred_min = torch.min(torch.topk(adj_rec, thred.int()).values)
        adj_rec = torch.where(adj_rec > thred_min, 1, 0)
        return adj_rec
