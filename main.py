from arguments import get_args
from live_video import LiveVideo
import numpy as np

import dgl
import torch
import networkx as nx
import matplotlib.pyplot as plt


def build_graph(node_num, edge_list):
    g = dgl.DGLGraph()
    # add 34 nodes into the graph; nodes are labeled from 0~33
    g.add_nodes(node_num)
    # add edges two lists of nodes: src and dst
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    # edges are directional in DGL; make them bi-directional
    g.add_edges(dst, src)

    return g

if __name__ == '__main__':

    args = get_args()

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
    historyRecord = []

    # 主循环
    for a in range(totalTime):
        # edge list for GCN
        edge_list = []
        # 获取所有用户历史记录
        for index, client in enumerate(videoUsers):
            historyRecord.append(client.get_history())

        for index1, value1 in enumerate(historyRecord):
            for index2, value2 in enumerate(historyRecord[index1 + 1:]):
                similarity = np.sum(np.trunc((value1 + value2)) != 1)
                if similarity > args.threshold:
                    edge_list.append((index1, index2 + index1 + 1))

        graph = build_graph(totalUser, edge_list)

        # 画个图看看情况
        print('%d nodes.' % graph.number_of_nodes())
        print('%d edges.' % graph.number_of_edges())

        fig, ax = plt.subplots()
        fig.set_tight_layout(False)
        nx_G = graph.to_networkx().to_undirected()
        pos = nx.kamada_kawai_layout(nx_G)
        nx.draw(nx_G, pos, with_labels=True, node_color=[[0.5, 0.5, 0.5]])
        plt.show()

        # assign features to nodes or edges
        graph.ndata['feat'] = torch.eye(34)
        print(graph.nodes[2].data['feat'])
        print(graph.nodes[1, 2].data['feat'])