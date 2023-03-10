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

    #print(hg)

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
        historyRecord = []
        futureRecord = []
        edge_list1 = []     # user to user
        edge_list3 = []  # tile to user
        labels = []
        FoV_list = []
        history_list = []
        pre_u_embeddings = []
        pre_t_embeddings = []

        for index1 in range(args.window * 2):
            for index2 in range(args.tileNum):
                tileTmp1 = index1 * 25 + index2 * 5
                for index3 in range(args.tileNum - 1):
                    edge_list3.append((tileTmp1 + index3, tileTmp1 + index3 + 1))
                edge_list3.append((tileTmp1, tileTmp1 + (args.tileNum - 1)))

            for index2 in range(args.tileNum):
                tileTmp2 = index1 * 25 + index2
                for index3 in range(args.tileNum - 1):
                    edge_list3.append((tileTmp2 + index3 * 5, tileTmp2 + (index3 + 1) * 5))
                edge_list3.append((tileTmp2, tileTmp2 + 5 * (args.tileNum - 1)))

        # 获取所有用户历史记录
        for index, client in enumerate(videoUsers):
            hist_vec, _ = client.get_history()
            historyRecord.append(hist_vec)

            next_vec, view_point_fix = client.get_nextView()
            if index == args.visId:
                view_point = view_point_fix

            vec_x = (view_point_fix[-1][0] - view_point_fix[0][0]) * 100
            vec_y = (view_point_fix[-1][1] - view_point_fix[0][1]) * 100
            distance = (vec_x ** 2 + vec_y ** 2) ** 0.5
            # pre_u_embeddings.append(hist_vec + [vec_x, vec_y, distance])
            pre_u_embeddings.append(hist_vec)
            pre_t_embeddings.append(next_vec)
            futureRecord.append(next_vec)

        for index1, value1 in enumerate(historyRecord):
            for index2, value2 in enumerate(historyRecord[index1 + 1:]):
                similarity = np.sum(np.trunc((np.sum([value1, value2], axis=0))) != 1)
                if similarity > args.threshold:
                    edge_list1.append((index1, index1 + 1 + index2))
            edge_list1.append((index1, index1))

        historyRecordNP = np.array(historyRecord) / (totalUser - args.testNum)
        his_vec.append(historyRecordNP.sum(axis=0))

        user_feats = th.tensor(pre_u_embeddings).to(th.float32).to('cuda:0')

        if iteration < 200:
            tile_feats = th.randn(args.tileNum ** 2 * args.window * 2, 200).to('cuda:0')
        else:
            futureNP = np.array(futureRecord[0:totalUser]).sum(axis=0) / args.testNum
            historyNP = his_vec[-199:]
            pre_t_embeddings = np.r_[historyNP, futureNP.reshape(1, 200)]
            tile_feats = th.tensor(pre_t_embeddings.T).to(th.float32).to('cuda:0')

        """
        for index1 in range(totalUser):
            if index1 >= args.testNum:
                # for index2, value2 in enumerate(historyRecord[index1]):
                #    if value2 == 1:
                #        edge_list2.append((index1, index2))
                for index2, value2 in enumerate(futureRecord[index1]):
                    if value2 == 1:
                        edge_list2.append((index1, index2))
            else:
                labels.append(futureRecord[index1])
        """

        for index1 in range(totalUser):
            labels.append(futureRecord[index1])

        edge_list2 = []  # tile to user
        for index1 in range(totalUser):
            for index2, value2 in enumerate(historyRecord[index1]):
                if value2 == 1:
                    edge_list2.append((index1, index2))

        k = 200

        for index1 in range(args.trainNum):
            for index2, value2 in enumerate(futureRecord[index1]):
                if value2 == 1:
                    edge_list2.append((index1, 200 + index2))

        # 补充位置关系
        # encoder = PosEncoder(200, dropout=0.4, max_len=200).cuda()
        # user_feats_en = encoder(user_feats)
        # tile_feats_en = encoder(tile_feats)

        for index1 in range(totalUser):

            if sum(labels[index1]) == 0:
                continue

            TP, TN, FP, FN = 0, 0, 0, 0
            PredictedTile = 0
            startT1 = time()
            '''
            for index2, value2 in enumerate(futureRecord[index1]):
                if value2 == 1:
                    edge_list2.append((args.trainNum + index1, 200 + index2))
            '''

            hGraph = build_graph(edge_list1, edge_list2, edge_list3, userEmbedding=user_feats, tileEmbedding=tile_feats)
            model = Model(200, 100, k, hGraph.etypes).cuda()
            user_feats = hGraph.nodes['user'].data['feature'].to('cuda:0')
            tile_feats = hGraph.nodes['tile'].data['feature'].to('cuda:0')
            node_features = {'user': user_feats, 'tile': tile_feats}
            opt = th.optim.Adam(model.parameters())

            for epoch in range(args.epochGCN):
                th.cuda.empty_cache()
                negative_graph = construct_negative_graph(hGraph, k, ('user', 'interest', 'tile'))
                pos_score, neg_score = model(hGraph, negative_graph, node_features, ('user', 'interest', 'tile'))
                loss = compute_loss(pos_score, neg_score)
                opt.zero_grad()
                loss.backward()
                opt.step()
                # print(loss.item())

            #
            # if index1 < args.trainNum:
            #    continue

            node_embeddings = model.sage(hGraph, node_features)

            user_embeddings = node_embeddings['user'][index1]
            tile_embeddings = node_embeddings['tile'][200:]

            result = model.predict(user_embeddings.reshape(1, k), tile_embeddings, thredhold[index1])

            if args.visId == index1:
                env.setPrediction(result[0, :])
                env.setFov(view_point)
                frames.append(env.render(mode='rgb_array'))
                env.render()

            for index2, value2 in enumerate(labels[index1]):
                if value2 == 1 and result[0, index2] == 1:   # result[index1, index2]
                    TP += 1
                    PredictedTile += 1
                elif value2 == 1 and result[0, index2] == 0:  # result[index1, index2]
                    FN += 1
                elif value2 == 0 and result[0, index2] == 1:  # result[index1, index2]
                    FP += 1
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

            if recall >= 0.9 and precision < 0.6:
                thredhold[index1] += -1
            elif recall < 0.9:
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