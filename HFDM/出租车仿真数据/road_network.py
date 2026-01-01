# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 13:00:36 2022

@author: Meng danyang
"""



import networkx as nx


# 创建路网
"""

input:
node_id ———— 点的编号
node_id_x & node_id_y ———— 边的两个端点
length ———— 边的长度
edge_id ———— 边的编号

output:
G ———— 生成的路网

"""




def network(node_id, node_id_x, node_id_y, length, edge_id):
    G = nx.Graph()  # 创建一个空图
    G.add_nodes_from(node_id)  # 添加点
    for i in range(len(node_id_x)):
        G.add_edge(node_id_x[i], node_id_y[i], weight=length[i], edge_id=edge_id[i])
    return G