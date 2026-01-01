# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 13:00:36 2022

@author: Meng danyang
"""

import random
import numpy as np
import networkx as nx

# 随机生成路径
"""
input:
G ———— 路网
node ———— 当前所在位置

output:
path ———— 随机生成的路径

"""

def path_random(G,node):
    path = []
    path.append(node)
    nei_node = list(G[node])  # 选出该节点的所有邻接节点
    tar_node = random.choice(nei_node)  # 从所有邻接节点中随机选一个点
    path.append(tar_node)
    return path


# 最短路径规划
"""
input:
G ———— 路网
start ———— 起点
end ———— 终点

output:
route ———— 最短路径

"""



def shortest_route_plan(G, start, end):
    route = nx.dijkstra_path(G, start, end)
    return route