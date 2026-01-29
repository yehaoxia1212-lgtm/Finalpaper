import numpy as np
from subfunction_network import convert_node_list_to_edge_list
from subfunction_network import generate_delta_paths_fast
from copy import deepcopy
import networkx as nx
import random


# 创建trip类
class trip_one_day:
    def __init__(self, no, start_station, start_time, end_station, end_time):
        self.no = no  # 行程编号
        self.start_station = start_station  # 开始站点
        self.start_time = start_time  # 离开时间
        self.end_station = end_station  # 结束站点
        self.end_time = end_time  # 到达时间
        # 每个trip可以覆盖到的路段
        self.trip_covered_edge = []  # 以列表的形式存储，路段编号

    def print_trip(self):
        print("trip编号:", self.no, end="  ")
        print("开始站点:", self.start_station, end="  ")
        print("离开时间:", self.start_time, end="  ")
        print("结束站点:", self.end_station, end="  ")
        print("到达时间:", self.end_time, end="  ")
        print("trip覆盖到的路段：", self.trip_covered_edge)
        print("trip覆盖到的路段数量:", len(self.trip_covered_edge))


# trip类的实例化
def trip_information_shortest_path(G, trip_data, adjacency_matrix):
    trip_list = []  # 存储所有的trip
    # 订单列表的实例化
    for i in range(len(trip_data)):
        start_station = int(trip_data.loc[i, "start_node_id"])
        end_station = int(trip_data.loc[i, "end_node_id"])
        start_time = trip_data.loc[i, "start_time"]
        end_time = trip_data.loc[i, "end_time"]
        trip_list.append(trip_one_day(i, start_station, start_time, end_station, end_time))
    for trip in trip_list:
        # 读取两个点之间的最短路列表
        shortest_path = nx.shortest_path(G, source=trip.start_station, target=trip.end_station, weight='length', method='dijkstra')
        # 把点的序列转化为边的序列
        edge_path_new = convert_node_list_to_edge_list(shortest_path, adjacency_matrix)
        trip.trip_covered_edge = deepcopy(edge_path_new)
    return trip_list


# trip类的实例化
def trip_information_delta_path(G, trip_data, adjacency_matrix, delta):
    trip_list = []  # 存储所有的trip
    # 订单列表的实例化
    for i in range(len(trip_data)):
        start_station = int(trip_data.loc[i, "start_node_id"])
        end_station = int(trip_data.loc[i, "end_node_id"])
        start_time = trip_data.loc[i, "start_time"]
        end_time = trip_data.loc[i, "end_time"]
        trip_list.append(trip_one_day(i, start_station, start_time, end_station, end_time))
    for trip in trip_list:
        # 读取两个点之间的路径
        print(trip.start_station, trip.end_station)
        new_route = generate_delta_paths_fast(G, trip.start_station, trip.end_station, delta, weight='weight', max_paths=5)
        delta_route_num = len(new_route)
        route_choose_no = random.randint(0, delta_route_num - 1)
        route_choose = new_route[route_choose_no][0]  # 最终选择的路线
        print("最终选择的路径长度为：", new_route[route_choose_no][1])
        # 把点的序列转化为边的序列
        edge_path_new = convert_node_list_to_edge_list(route_choose, adjacency_matrix)
        trip.trip_covered_edge = deepcopy(edge_path_new)
    return trip_list