from arguments import get_args
from live_video import LiveVideo
import numpy as np

import torch as th
from model.EVOLVE_GCN import EvolveGCNO, EvolveGCNH
from matplotlib import animation
import matplotlib.pyplot as plt
import gym
from time import time
import networkx as nx
from model.random_walk import Graph_RandomWalk
from collections import defaultdict

import dgl


def display_frames_as_gif(policy, frames):
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=5)
    anim.save('./' + policy + '_viewport_result.gif', writer='ffmpeg', fps=30)


def build_graph(edge_list, nodeFeature):

    src, dst = tuple(zip(*edge_list))
    src, dst = th.tensor(src).to('cuda:0'), th.tensor(dst).to('cuda:0')
    u = th.cat((src, dst))
    v = th.cat((dst, src))

    g = dgl.graph((u, v))
    g.ndata['feat'] = nodeFeature
    #print(g)

    return g


def compute_loss(pos_score, neg_score):
    # 间隔损失
    n_edges = pos_score.shape[0]
    return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()


def construct_negative_graph(graph, k):
    src, dst = graph.edges()

    neg_src = src.repeat_interleave(k).to('cuda:0')
    neg_dst = th.randint(0, graph.num_nodes(), (len(src) * k,)).to('cuda:0')
    return dgl.graph((neg_src, neg_dst), num_nodes=graph.num_nodes())


def run_random_walks_n2v(graph, nodes, num_walks=10, walk_len=40):
    """ In: Graph and list of nodes
        Out: (target, context) pairs from random walk sampling using the sampling strategy of node2vec (deepwalk)"""
    walk_len = 5
    nx_G = nx.Graph()
    adj = nx.adjacency_matrix(graph)
    for e in graph.edges():
        nx_G.add_edge(e[0], e[1])

    for edge in graph.edges():
        nx_G[edge[0]][edge[1]]['weight'] = adj[edge[0], edge[1]]

    G = Graph_RandomWalk(nx_G, False, 1.0, 1.0)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks, walk_len)
    WINDOW_SIZE = 10
    pairs = defaultdict(lambda: [])
    pairs_cnt = 0
    for walk in walks:
        for word_index, word in enumerate(walk):
            for nb_word in walk[max(word_index - WINDOW_SIZE, 0): min(word_index + WINDOW_SIZE, len(walk)) + 1]:
                if nb_word != word:
                    pairs[word].append(nb_word)
                    pairs_cnt += 1
    print("# nodes with random walk samples: {}".format(len(pairs)))
    print("# sampled pairs: {}".format(pairs_cnt))
    return pairs


def get_context_pairs(graphs, num_time_steps):
    """ Load/generate context pairs for each snapshot through random walk sampling."""
    print("Computing training pairs ...")
    context_pairs_train = []
    for i in range(0, num_time_steps):
        context_pairs_train.append(run_random_walks_n2v(graphs[i], graphs[i].nodes()))

    return context_pairs_train


if __name__ == '__main__':

    args = get_args()     # 从 arguments.py 获取配置参数

    # graphs, adjs, labels = load_graphs_school('school')
    #

    if args.cuda and th.cuda.is_available():
        device = th.device('cuda:0')
        th.backends.cudnn.benchmark = True
    else:
        device = th.device('cpu')

    videoUsers = []

    tileNum = args.sampleRate * args.tileNum * args.tileNum   # 瓦块数：一个视频帧切分成 5 * 5 = 25 个瓦块

    totalUser = args.testNum + args.trainNum

    # 生成所有用户的视频信息类
    for index in range(totalUser):
        args.userId = index + 1
        videoUsers.append(LiveVideo(args))

    # 为每个用户加载视频数据
    for index in range(totalUser):
        videoUsers[index].videoLoad()

    totalTime = videoUsers[0].get_time()

    # 主循环
    env = gym.make('MyEnv-v1')
    frames = []
    his_vec = []
    thredhold = args.thred * th.ones(args.testNum + args.trainNum)

    for iteration in range(totalTime):
        # edge list for GCN
        historyRecord = []      # 用户行为的历史记录
        futureRecord = []       # 用户行为未来记录

        node_list = []          # 用户和瓦片的节点列表

        edge_list_p = []        # 正采样图神经网络
        edge_list_n = []        # 负采样图神经网络

        his_and_fut_list = []   # 历史和未来图关系整合

        labels = []             # 真实的观看记录标签
        pre_u_embeddings = []   # 初始用户的embeddings
        pre_t_embeddings = []   # 初始瓦片的embeddings
        node_feature = []

        # 获取所有用户历史记录
        for index, client in enumerate(videoUsers):
            # 获取历史的观看记录
            hist_vec, _ = client.get_history()
            historyRecord.append(hist_vec)

            # 获取未来的观看记录
            next_vec, view_point_fix = client.get_nextView()
            futureRecord.append(next_vec)

            # 获取综合观看记录
            total_vec = hist_vec + next_vec
            total_vec = np.array(total_vec).astype(np.int32)
            his_and_fut_list.append(total_vec.reshape(2 * args.window, args.tileNum ** 2))
            labels.append(np.array(next_vec).reshape(args.window, args.tileNum ** 2))

            # 可视化选项
            if index == args.visId:
                view_point = view_point_fix

            vec_x = (view_point_fix[-1][0] - view_point_fix[0][0]) * 100
            vec_y = (view_point_fix[-1][1] - view_point_fix[0][1]) * 100
            distance = (vec_x ** 2 + vec_y ** 2) ** 0.5

            # 初始的用户embeddings设置
            pre_u_embeddings.append(hist_vec)

        # 构建用户和瓦片的节点列表
        n_node = totalUser + args.tileNum ** 2
        for index1 in range(args.window * 2):
            node_tmp = []
            for index1 in range(n_node):
                node_tmp.append((index1, {}))
            node_list.append(node_tmp)

        for index1 in range(args.tileNum ** 2):
            pre_u_embeddings.append(np.random.rand(200).tolist())

        # 计算每帧用户的相似关系
        for index1 in range(args.window * 2):
            edge_tmp = []
            for index2, value2 in enumerate(his_and_fut_list):
                for index3, value3 in enumerate(his_and_fut_list[index2 + 1:]):
                    similarity = np.sum(np.trunc((np.sum([value3[index1], value2[index1]], axis=0))) != 1)
                    if similarity > args.threshold:
                        edge_tmp.append((index2, index2 + 1 + index3))

            # 构建用户观看瓦片的关系
            for index2, value2 in enumerate(his_and_fut_list):
                for index3, value3 in enumerate(value2[index1]):
                    if value3 == 1:
                        edge_tmp.append((index2, totalUser + index3))

            # 构建瓦片与瓦片之间的位置关系图，相邻的瓦片存在相似关系
            for index2 in range(args.tileNum):
                tileTmp1 = index2 * 5
                for index3 in range(args.tileNum - 1):
                    edge_tmp.append((totalUser + tileTmp1 + index3,
                                     totalUser + tileTmp1 + index3 + 1))

            for index2 in range(args.tileNum):
                tileTmp2 = index2
                for index3 in range(args.tileNum - 1):
                    edge_tmp.append((totalUser + tileTmp2 + index3 * 5,
                                     totalUser + tileTmp2 + (index3 + 1) * 5))

            edge_list_p.append(edge_tmp)
            node_feature.append(pre_u_embeddings)

        futureRecord = np.array(futureRecord)

        '''
        for index1 in range(totalUser):
            label_one = []
            for index2 in range(args.window):
                label_one.append(futureRecord[index1][index2::args.window])
            labels.append(label_one)
        '''

        k = args.input_dim
        node_feature = th.tensor(node_feature).to(th.float32).to('cuda:0')

        TP, TN, FP, FN = 0, 0, 0, 0
        PredictedTile = 0
        startT1 = time()

        for index1 in range(totalUser - 1):

            n_graphs = []
            graphs = []
            for index2 in range(args.window * 2):
                Graph = build_graph(edge_list_p[index2],
                                    node_feature[index2])
                nGraph = construct_negative_graph(Graph, 1)
                nGraph.ndata['feat'] = node_feature[index2]
                graphs.append(Graph)
                n_graphs.append(nGraph)

            if args.model == 'EvolveGCN-O':
                model = EvolveGCNO(in_feats=k,
                                   n_hidden=args.n_hidden,
                                   num_layers=args.n_layers,
                                   n_classes=args.n_output)
            elif args.model == 'EvolveGCN-H':
                model = EvolveGCNH(in_feats=k,
                                   num_layers=args.n_layers)

            model = model.to(device)
            optimizer = th.optim.Adam(model.parameters(), lr=args.lr)

            for i in range(args.window, args.window * 2 - 1):
                g_list = graphs[i - args.window:i]
                ng_list = n_graphs[i - args.window:i]
                for epoch in range(args.epochGCN):
                    model.train()
                    # get predictions which has label
                    pos_score, neg_score = model(g_list, ng_list, node_feature[i - args.window:i])
                    loss = compute_loss(pos_score, neg_score)

                    optimizer.zero_grad()
                    loss.backward()
                    print(loss)
                    optimizer.step()

                # 设置端点处
                node_embeddings = model.saga(graphs[i - args.window:i + 1], node_feature[i - args.window:i + 1])
                user_embeddings = node_embeddings[0:totalUser]
                tile_embeddings = node_embeddings[totalUser - 1:-1]
                thredhold[index1] = sum(his_and_fut_list[index1][i]) * 2
                if thredhold[index1] == 0:
                    thredhold[index1] = 1
                result = model.predict(user_embeddings[index1].reshape(1, args.n_output), tile_embeddings, thredhold[index1])
                for index2, value2 in enumerate(his_and_fut_list[index1][i]):
                    if value2 == 1 and result[0, index2] == 1:    # result[index1, index2]
                        TP += 1
                        PredictedTile += 1
                    elif value2 == 1 and result[0, index2] == 0:  # result[index1, index2]
                        FP += 1
                    elif value2 == 0 and result[0, index2] == 1:  # result[index1, index2]
                        FN += 1
                        PredictedTile += 1
                    elif value2 == 0 and result[0, index2] == 0:  # result[index1, index2]
                        TN += 1
            """
                for index, node_one in enumerate(node_embeddings):
                    user_embeddings = node_one[0:totalUser]
                    tile_embeddings = node_one[totalUser - 1:-1]
                    result = model.predict(user_embeddings[index1].reshape(1, k), tile_embeddings, thredhold[index1])
                    for index2, value2 in enumerate(labels[index1][index]):
                        if value2 == 1 and result[0, index2] == 1:   # result[index1, index2]
                            TP += 1
                            PredictedTile += 1
                        elif value2 == 1 and result[0, index2] == 0:  # result[index1, index2]
                            FP += 1
                        elif value2 == 0 and result[0, index2] == 1:  # result[index1, index2]
                            FN += 1
                            PredictedTile += 1
                        elif value2 == 0 and result[0, index2] == 0:  # result[index1, index2]
                            TN += 1
            """

        endT1 = time()
        totalT1 = endT1 - startT1

        if TP + TN + FP + FN == 0:
            accuracy = 0
        else:
            accuracy = (TP + TN) / (TP + TN + FP + FN)

        if (TP + FP) == 0:
            precision = 0
        else:
            precision = TP / (TP + FP)

        if (TP + FN) == 0:
            recall = 0
        else:
            recall = TP / (TP + FN)

        avePreTile = PredictedTile / 8  # / (8 * args.testNum)

        if precision >= 0.8 and recall < 0.6:
            thredhold[index1] += -1
        elif precision < 0.8:
            thredhold[index1] += +1
        aveTime = totalT1  # / (args.testNum + args.trainNum)

        print("accuracy:", str(accuracy),
                 "precision:", str(precision),
                 "recall:", str(recall),
                 "predicted tile:", str(avePreTile),
                 "Train Time:", str(aveTime))

        sortedValues = [str(accuracy), str(precision), str(recall), str(avePreTile), str(aveTime)]
        videoUsers[index1].allWriter.writerCSVA(sortedValues)

    display_frames_as_gif(args.policy, frames)