from arguments import get_args
from live_video import LiveVideo
import numpy as np

import dgl
import torch as th
from model.RGCN import Model
from matplotlib import animation
import matplotlib.pyplot as plt
import gym


def display_frames_as_gif(policy, frames):
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=5)
    anim.save('./' + policy + '_viewport_result.gif', writer='ffmpeg', fps=30)


def build_graph(edge_list1, edge_list2, edge_list3):

    src1, dst1 = tuple(zip(*edge_list1))
    src1, dst1 = th.tensor(src1), th.tensor(dst1)
    u = th.cat((src1, dst1))
    v = th.cat((dst1, src1))

    src2, dst2 = tuple(zip(*edge_list2))
    src2, dst2 = th.tensor(src2), th.tensor(dst2)
    w = src2
    x = dst2

    src3, dst3 = tuple(zip(*edge_list3))
    src3, dst3 = th.tensor(src3), th.tensor(dst3)
    y = th.cat((src3, dst3))
    z = th.cat((dst3, src3))
    # Create a heterograph with 2 node types and 2 edges types.
    graph_data = {
        ('user', 'similarity', 'user'): (u, v),
        ('user', 'interest', 'tile'): (w, x),
        ('tile', 'with', 'tile'): (y, z)
    }
    hg = dgl.heterograph(graph_data)

    hg.nodes['user'].data['feature'] = th.randn(hg.num_nodes('user'), 10)
    hg.nodes['tile'].data['feature'] = th.randn(hg.num_nodes('tile'), 10)
    hg.edges['similarity'].data['weight'] = th.ones(hg.num_edges('similarity'), 1)
    hg.edges['interest'].data['weight'] = th.ones(hg.num_edges('interest'), 1)

    print(hg)

    # g = dgl.graph((u, v), num_nodes=user_num + tile_num)
    return hg


def tuple_of_tensors_to_tensor(tuple_of_tensors):
    return th.stack(list(tuple_of_tensors), dim=0)


def construct_negative_graph(graph, k, etype):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    neg_dst = th.randint(0, graph.num_nodes(vtype), (len(src) * k,))
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})


def compute_loss(pos_score, neg_score):
    # 间隔损失
    n_edges = pos_score.shape[0]
    return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()


if __name__ == '__main__':

    args = get_args()

    if args.cuda and th.cuda.is_available():
        device = th.device('cuda')
        th.backends.cudnn.benchmark = True
    else:
        device = th.device('cpu')

    videoUsers = []

    tileNum = args.sampleRate * 5 * 5

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

    for a in range(totalTime):
        # edge list for GCN
        historyRecord = []
        edge_list1 = []     # user to user
        edge_list2 = []     # tile to user
        edge_list3 = []  # tile to user
        labels = []

        '''
        for index1 in range(0, 199, 5):
            for index2 in range(4):
                edge_list3.append((index1 + index2, index1 + index2 + 1))
        '''
        edge_list3.append((0, 199))

        # 获取所有用户历史记录
        for index, client in enumerate(videoUsers):
            next_vec, view_point_fix = client.get_history()
            if index == args.visId + args.trainNum:
                view_point = view_point_fix
            historyRecord.append(next_vec)

        for index1, value1 in enumerate(historyRecord):
            for index2, value2 in enumerate(historyRecord[index1 + 1:]):
                similarity = np.sum(np.trunc((np.sum([value1, value2], axis=0))) != 1)
                if similarity > args.threshold:
                    edge_list1.append((index1, index2))

        for index1 in range(totalUser):
            if index1 < args.trainNum:
                for index2, value2 in enumerate(historyRecord[index1]):
                    if value2 == 1:
                        edge_list2.append((index1, index2))
            labels.append(historyRecord[index1])

        hGraph = build_graph(edge_list1, edge_list2, edge_list3)

        k = 5
        model = Model(10, 20, k, hGraph.etypes)
        user_feats = hGraph.nodes['user'].data['feature']
        tile_feats = hGraph.nodes['tile'].data['feature']
        node_features = {'user': user_feats, 'tile': tile_feats}
        opt = th.optim.Adam(model.parameters())
        for epoch in range(args.epochGCN):
            negative_graph = construct_negative_graph(hGraph, k, ('user', 'interest', 'tile'))
            pos_score, neg_score = model(hGraph, negative_graph, node_features, ('user', 'interest', 'tile'))
            loss = compute_loss(pos_score, neg_score)
            opt.zero_grad()
            loss.backward()
            opt.step()
            # print(loss.item())

        node_embeddings = model.sage(hGraph, node_features)

        user_embeddings = node_embeddings['user'][0:args.trainNum]
        tile_embeddings = node_embeddings['tile']

        result = model.predict(user_embeddings, tile_embeddings, args.thred)

        TP, TN, FP, FN = 0, 0, 0, 0
        PredictedTile = 0
        for index1, value1 in enumerate(labels[args.trainNum:-1]):

            if args.visId == index1:
                env.setPrediction(result[index1, :])
                env.setFov(view_point)
                frames.append(env.render(mode='rgb_array'))
                env.render()

            for index2, value2 in enumerate(value1):
                if value2 == 1 and result[index1, index2] == 1:
                    TP += 1
                    PredictedTile += 1
                elif value2 == 1 and result[index1, index2] == 0:
                    FP += 1
                elif value2 == 0 and result[index1, index2] == 1:
                    FN += 1
                    PredictedTile += 1
                elif value2 == 0 and result[index1, index2] == 0:
                    TN += 1

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        if (TP + FP) == 0:
            precision = 0
        else:
            precision = TP / (TP + FP)

        if (TP + FN) == 0:
            recall = 0
        else:
            recall = TP / (TP + FN)

        avePreTile = PredictedTile / (8 * args.testNum)

        if precision >= 0.8 and recall < 0.6:
            args.thred += +0.05
        elif precision < 0.8:
            args.thred += -0.05

        print("accuracy:", str(accuracy),
              "precision:", str(precision),
              "recall", str(recall),
              "predicted tile", str(avePreTile))

    display_frames_as_gif(args.policy, frames)