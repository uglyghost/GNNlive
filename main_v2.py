from arguments import get_args
from live_video import LiveVideo
import numpy as np

import dgl
import torch as th
from model.RGCN import Model
from model.PositionalEncoding import PositionalEncoding as PosEncoder
from matplotlib import animation
import matplotlib.pyplot as plt
import gym
from time import time


def display_frames_as_gif(policy, frames):
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=5)
    anim.save('./' + policy + '_viewport_result.gif', writer='ffmpeg', fps=30)


def build_graph(edge_list1, edge_list2, edge_list3, userEmbedding, tileEmbedding):

    src1, dst1 = tuple(zip(*edge_list1))
    src1, dst1 = th.tensor(src1).to('cuda:0'), th.tensor(dst1).to('cuda:0')
    u = th.cat((src1, dst1))
    v = th.cat((dst1, src1))

    src2, dst2 = tuple(zip(*edge_list2))
    src2, dst2 = th.tensor(src2).to('cuda:0'), th.tensor(dst2).to('cuda:0')
    w = src2
    x = dst2

    src3, dst3 = tuple(zip(*edge_list3))
    src3, dst3 = th.tensor(src3).to('cuda:0'), th.tensor(dst3).to('cuda:0')
    y = th.cat((src3, dst3))
    z = th.cat((dst3, src3))
    # Create a heterograph with 2 node types and 2 edges types.
    graph_data = {
        ('user', 'similarity', 'user'): (u, v),
        ('user', 'interest', 'tile'): (w, x),
        ('tile', 'with', 'tile'): (y, z)
    }
    hg = dgl.heterograph(graph_data)

    hg.nodes['user'].data['feature'] = userEmbedding
    hg.nodes['tile'].data['feature'] = tileEmbedding
    hg.edges['similarity'].data['weight'] = 0.2 * th.ones(hg.num_edges('similarity'), 1).to('cuda:0')
    hg.edges['interest'].data['weight'] = th.ones(hg.num_edges('interest'), 1).to('cuda:0')
    hg.edges['with'].data['weight'] = 0.2 * th.ones(hg.num_edges('with'), 1).to('cuda:0')

    print(hg)

    # g = dgl.graph((u, v), num_nodes=user_num + tile_num)
    return hg


def tuple_of_tensors_to_tensor(tuple_of_tensors):
    return th.stack(list(tuple_of_tensors), dim=0)


def construct_negative_graph(graph, k, etype):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k).to('cuda:0')
    neg_dst = th.randint(0, graph.num_nodes(vtype), (len(src) * k,)).to('cuda:0')
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})


def compute_loss(pos_score, neg_score):
    # 间隔损失
    n_edges = pos_score.shape[0]
    return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()


if __name__ == '__main__':

    args = get_args()     # 从 arguments.py 获取配置参数

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

        edge_list1 = []         # 用户和用户的相似关系
        edge_list2 = []         # 用户和视频瓦片的关系
        edge_list3 = []         # 瓦片与瓦片之间的关系

        his_and_fut_list = []   # 历史和未来图关系整合

        labels = []             # 真实的观看记录标签
        pre_u_embeddings = []   # 初始用户的embeddings
        pre_t_embeddings = []   # 初始瓦片的embeddings

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
            his_and_fut_list.append(np.array(total_vec).reshape(2 * args.window, args.tileNum ** 2))

            # 可视化选项
            if index == args.visId:
                view_point = view_point_fix

            vec_x = (view_point_fix[-1][0] - view_point_fix[0][0]) * 100
            vec_y = (view_point_fix[-1][1] - view_point_fix[0][1]) * 100
            distance = (vec_x ** 2 + vec_y ** 2) ** 0.5

            # 初始的用户embeddings设置
            pre_u_embeddings.append(hist_vec)

        # 计算每帧用户的相似关系
        for index1 in range(args.window * 2):
            edge_tmp = []
            for index2, value2 in enumerate(his_and_fut_list):
                for index3, value3 in enumerate(his_and_fut_list[index2 + 1:]):
                    similarity = np.sum(np.trunc((np.sum([value3[index1], value2[index1]], axis=0))) != 1)
                    if similarity > args.threshold:
                        edge_tmp.append((index2, index2 + 1 + index3))
            edge_list1.append(edge_tmp)

        # 构建用户观看瓦片的关系
        for index1 in range(args.window * 2):
            edge_tmp = []
            for index2, value2 in enumerate(his_and_fut_list):
                for index3, value3 in enumerate(value2[index1]):
                    if value3 == 1:
                        edge_tmp.append((index2, index3))
            edge_list2.append(edge_tmp)

        # 构建瓦片与瓦片之间的位置关系图，相邻的瓦片存在相似关系
        for index1 in range(args.window * 2):
            edge_tmp = []
            for index2 in range(args.tileNum):
                tileTmp1 = index2 * 5
                for index3 in range(args.tileNum - 1):
                    edge_tmp.append((tileTmp1 + index3, tileTmp1 + index3 + 1))

            for index2 in range(args.tileNum):
                tileTmp2 = index2
                for index3 in range(args.tileNum - 1):
                    edge_tmp.append((tileTmp2 + index3 * 5, tileTmp2 + (index3 + 1) * 5))
            edge_list3.append(edge_tmp)

        # 瓦片的初始化特征为流行度变化趋势
        historyRecordNP = np.array(historyRecord) / (totalUser - args.testNum)
        his_vec.append(historyRecordNP.sum(axis=0))

        user_feats = th.tensor(pre_u_embeddings).to(th.float32).to('cuda:0')

        if iteration < args.input_dim:
            # 当记录不足 input_dim 条，随机初始化瓦片特征
            tile_feats = th.randn(args.tileNum ** 2 * args.window, args.input_dim).to('cuda:0')
        else:
            futureNP = np.array(futureRecord[0:args.testNum]).sum(axis=0) / args.testNum
            historyNP = his_vec[1 - args.input_dim:]
            pre_t_embeddings = np.r_[historyNP, futureNP.reshape(1, args.input_dim)]
            tile_feats = th.tensor(pre_t_embeddings.T).to(th.float32).to('cuda:0')

        for index1 in range(totalUser):
            labels.append(futureRecord[index1])

        k = args.input_dim

        for index1 in range(totalUser - 1):
            TP, TN, FP, FN = 0, 0, 0, 0
            PredictedTile = 0
            startT1 = time()

            hGraph = build_graph(edge_list1, edge_list2, edge_list3, userEmbedding=user_feats, tileEmbedding=tile_feats)

            # 等待修改
            model = Model(args.input_dim, 100, k, hGraph.etypes).cuda()
            user_feats = hGraph.nodes['user'].data['feature'].to('cuda:0')
            tile_feats = hGraph.nodes['tile'].data['feature'].to('cuda:0')
            node_features = {'user': user_feats, 'tile': tile_feats}
            opt = th.optim.Adam(model.parameters())

            for epoch in range(args.epochGCN):
                negative_graph = construct_negative_graph(hGraph, k, ('user', 'interest', 'tile'))
                pos_score, neg_score = model(hGraph, negative_graph, node_features, ('user', 'interest', 'tile'))
                node_embeddings = model.sage(hGraph, node_features)
                loss = compute_loss(pos_score, neg_score)
                opt.zero_grad()
                loss.backward()
                opt.step()
            th.cuda.empty_cache()
            # print(loss.item())

            node_embeddings = model.sage(hGraph, node_features)

            user_embeddings = node_embeddings['user'][index1 + 1]
            tile_embeddings = node_embeddings['tile']

            result = model.predict(user_embeddings.reshape(1, k), tile_embeddings, thredhold[index1])

            if args.visId == index1 + 1:
                env.setPrediction(result[0, :])
                env.setFov(view_point)
                frames.append(env.render(mode='rgb_array'))
                env.render()

            for index2, value2 in enumerate(labels[index1 + 1]):
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