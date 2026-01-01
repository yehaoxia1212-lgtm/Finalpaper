# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 13:00:36 2022

@author: Meng danyang
"""


import pandas as pd
import networkx as nx
import numpy as np

from route_plan import shortest_route_plan  #导入最短路径规划的函数


# 订单的初始化
"""
input:
G ———— 路网
order_list1 ———— 生成的订单列表
order ———— 订单的类

output:
order_list ———— 初始化后的订单

"""

def order_ini(G, order, order_list1):
    order_list = []
    for i in range(len(order_list1)):
        route1 = shortest_route_plan(G, order_list1.iloc[i]['node_on'], order_list1.iloc[i]['node_off'])  # 生成每个订单起终点之间的最短路
        order_list.append(order(order_list1.iloc[i]['no'], order_list1.iloc[i]['node_on'], order_list1.iloc[i]['node_off'], 1, 0, route1))
        # 订单的初始状态为1：未匹配，保留时间为0，以后每保留一个dt就+1，大于3了就删除，最短路径为[]
    return order_list