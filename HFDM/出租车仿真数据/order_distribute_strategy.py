# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 13:00:36 2022

@author: Meng danyang
"""



import random
import numpy as np
import networkx as nx
from route_plan import shortest_route_plan  #导入最短路径规划的函数


#订单分配策略
"""
input:
G ———— 路网
passenger_pos ———— 待服务的订单位置
taxi_pos ———— 空闲出租车的位置

output:
match ———— 匹配结果
path_infos ———— 各个出租车到各个乘客的路径信息
"""


def order_dis_strategy (G, passenger_pos,taxi_pos):

    # 保存各个出租车到各个乘客的路径信息
    path_infos = [[{} for j in range(len(passenger_pos))] for i in range(len(taxi_pos))]
    match = []
    for i in range(len(taxi_pos)): #对于每一辆车
        path_dis = 10000000000  #初始给定一个无限大的距离
        match_order = -1  #初始给定-1作为匹配订单，认为还未被匹配
        for j in range(len(passenger_pos)): #保存每一个乘客到该出租车的路径信息
            path = shortest_route_plan(G, taxi_pos[i], passenger_pos[j])
            dist = nx.dijkstra_path_length(G, taxi_pos[i], passenger_pos[j])
            path_infos[i][j]['p'] = path
            path_infos[i][j]['d'] = dist
            if j not in match:  #如果这个订单还未被匹配过
                if dist < path_dis:  #如果距离比记录的小
                    path_dis = dist
                    match_order = j
        match.append(match_order)

    return match, path_infos