import pandas as pd
import os
from gurobipy import *
import pickle
from subfunction_optimization import sensor_dis_model_one_month
import numpy as np
from collections import defaultdict
import random
import json


class Vehicle:
    def __init__(self, no, current_station):
        self.no = no
        self.current_station = current_station
        self.traj = {}  # 按天存储轨迹，{day: [...]}

    def print_vehicle(self):
        print("车辆编号:", self.no,
              "当前所在站点:", self.current_station,
              "车辆轨迹:", self.traj)


# 构建一个函数找到vehicle_list中当前站点编号等于某个特定值的车辆编号列表
def find_vehicle_according_station(vehicle_list, nnn):
    vehicle_no_list_temp = []
    for iii in vehicle_list:
        if iii.current_station == nnn:
            vehicle_no_list_temp.append(iii.no)
    return vehicle_no_list_temp


def save_trajectories(vehicle_list, out_path):
    """
    将 vehicle_list 中每辆车的 traj 字典写成一个 JSON 文件
    结构: {vehicle_no: {day: [trip1, trip2, ...]}}
    """
    traj_dict = {
        v.no: v.traj  # {day: [...]} 已按天存储
        for v in vehicle_list
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(traj_dict, f, ensure_ascii=False, indent=2)


# 读取基础数据
# 读取所有的站点数据
station_data = pd.read_csv("data/数据清洗之后的单车数据/每天开始时刻各站点所需的车辆数.csv")
station_id_list = station_data["station_id"].tolist()
station_id_list = [int(i) for i in station_id_list]

start_vehicle_num_data = pd.read_csv("data/数据清洗之后的单车数据/每天开始时刻各站点所需的车辆数.csv")
# 存储每天开始时刻各个站点的共享单车的数量字典,格式（日期，站点）：车辆数
start_vehicle_num_every_day_dic = {}
# 遍历日期
for no in range(1, 32):
    # 遍历每一行数据
    for _, row in start_vehicle_num_data.iterrows():
        station_id = int(row['station_id'])
        fleet_size = row["day_" + str(no)]
        # 将站点需求添加到当天的字典中
        start_vehicle_num_every_day_dic[(no, station_id)] = fleet_size

# 存储各路段长度字典
edge_length_dic = {}
edge_data = pd.read_excel("data/Manhattan_bike_road_network/edge.xls")
for _, row in edge_data.iterrows():
    edge_id = int(row['edge_id'])
    length = row["length"] / 1000
    edge_length_dic[edge_id] = length
edge_list = edge_data["edge_id"].tolist()

# 读取二项分布参数
k_value_dic = {}
for s in station_id_list:
    # 读取这个站点的二项分布参数
    k_data = pd.read_csv("data/二项分布/station_k_values/" + str(s) + "_k_values.csv")
    for _, row in k_data.iterrows():
        edge_id = int(row['edge_id'])
        k_value = row["k_value"]
        k_value_dic[(s, edge_id)] = k_value

# 日期列表
total_days = 31
all_days = list(range(1, total_days + 1))  # 1-31日
days_from_2 = list(range(2, total_days + 1))  # 2-31日
dispatch_days = list(range(1, total_days))  # 1-30日（调度日）

N_k = 1000

# 读取第一天的传感器分配数据
sensor_num_first_day_data = pd.read_csv("data/传感器布设优化/First_day/sensor_first_day_" + str(N_k) + ".csv")
sensor_num_first_day_dic = {}
for _, row in sensor_num_first_day_data.iterrows():
    station_id = row["Sensor"]
    count = row["Count"]
    sensor_num_first_day_dic[station_id] = count

# 根据sensor_num_first_day_dic选择第一天的轨迹，初始化第一天的轨迹
# 初始化第一天的轨迹
day = 1
base_dir = "data/数据清洗之后的单车数据/trip_data_march_every_day_6_20_new"
# 动态构建文件名
date_str = f'2024-03-{day:02d}'
filename = f"trip_data_{date_str}.csv"
file_path = os.path.join(base_dir, filename)
# 读取这一天的订单数据
trip_data = pd.read_csv(file_path)
trip_id_start_node_id_pair_dic = {}  # 存储trip和其对应的开始站点
trip_id_end_node_id_pair_dic = {}  # 存储trip和其对应的结束站点
for jjj in range(len(trip_data)):
    trip_id_start_node_id_pair_dic[jjj] = int(trip_data.loc[jjj, "start_node_id"])
    trip_id_end_node_id_pair_dic[jjj] = int(trip_data.loc[jjj, "end_node_id"])
# 读取第一天的轨迹数据
with open(f"data/数据清洗之后的单车数据/最短路随机车辆轨迹/" + str(day) + "_route.json", 'r') as f:
    vehicle_route_data = json.load(f)

empty_route_no_list = []
for key, value in vehicle_route_data.items():
    if len(value) == 0:
        empty_route_no_list.append(key)

# 实例化第一天的车辆列表
vehicle_list = []
no = 0
for station, sensor_num in sensor_num_first_day_dic.items():
    if int(sensor_num) == 0:
        continue
    else:
        # 读取第day天开始时刻站点station的车辆数
        all_num = int(start_vehicle_num_every_day_dic[(day, station)])
        route_no_list = []
        # 从vehicle_route_data中找到所有从该站点出发的路线编号
        for key, value in vehicle_route_data.items():
            if len(value) != 0:
                if trip_id_start_node_id_pair_dic[value[0][1]] == station:
                    route_no_list.append(key)
        # 看一下现在找到的这个站点的车辆数
        num = len(route_no_list)
        gap = all_num - num
        if gap != 0:
            # 把empty_route_no_list中前gap辆车的站点改掉
            choose_empty_route_no_list = empty_route_no_list[:gap]
            del empty_route_no_list[:gap]  # 删除
        else:
            choose_empty_route_no_list = []
        all_route_no_list = route_no_list + choose_empty_route_no_list
        if len(all_route_no_list) != all_num:
            print("路线查找存在问题")
        # 从all_route_no_list中随机选择sensor_num条轨迹，用来实例化
        choose_result_list = random.sample(all_route_no_list, int(sensor_num))
        for w in choose_result_list:
            if len(vehicle_route_data[w]) == 0:
                v = Vehicle(no, station)
                vehicle_list.append(v)
                v.traj[1] = []
            else:
                v = Vehicle(no, trip_id_end_node_id_pair_dic[vehicle_route_data[w][-1][1]])
                vehicle_list.append(v)
                v.traj[1] = vehicle_route_data[w]
            no += 1

# 在确定了第一天的传感器分配数据之后，day-to-day的决策每天的调度计划
for day in dispatch_days:
    print("##########################################################################正在运行第day天的数据：", day)
    # 经过一天的运行得到这一天结束时候各站点的车辆数
    sensor_num_end_day_dic = defaultdict(int)       # 初始化
    for vehicle in vehicle_list:
        sensor_num_end_day_dic[int(vehicle.current_station)] += 1

    # 根据vehicle_list计算当前各路段的覆盖次数
    # 根据vehicle_list计算当前各路段的覆盖次数
    current_road_coverage_dic = defaultdict(int)  # 默认 0
    covered = set()  # 记录已出现过的路段
    for vehicle in vehicle_list:
        for route_list in vehicle.traj.values():
            if not route_list:
                continue
            for _, _, segments in route_list:
                for seg in segments:
                    e = int(seg)
                    if e not in covered:
                        current_road_coverage_dic[e] = 1
                        covered.add(e)

    # 以原本的调度计划为基础优化当天的调度计划并存储
    [sensor_num_first_day_dic, sensor_dispatch_dic] = sensor_dis_model_one_month(day, N_k, station_id_list, sensor_num_end_day_dic, edge_list, edge_length_dic, start_vehicle_num_every_day_dic, k_value_dic, current_road_coverage_dic)


    # 判断这一天更新之后的传感器数量是否还是等于N_k
    temp = 0
    for key, value in sensor_num_first_day_dic.items():
        temp += value
    if temp != N_k:
        print("！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！求解过程存在问题！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！")

    # 根据调度计划更新Vehicle_list的位置
    sensor_dispatch_nonzero_dic = {}  #格式（i,j）:调度车辆数
    for pair, value in sensor_dispatch_dic.items():
        if value > 0:
            sensor_dispatch_nonzero_dic[(pair[0], pair[1])] = value
    # 根据车辆调度计划更新车辆列表
    for key, value in sensor_dispatch_nonzero_dic.items():
        # 找到当前站点为key[0]的车辆编号
        find_vehicle_list = find_vehicle_according_station(vehicle_list, key[0])
        # 检查是否有足够车辆可调出
        if len(find_vehicle_list) >= value:
            # 随机选择要调出的车辆
            selected_vehicles = random.sample(find_vehicle_list, int(value))
            # 更新这些车辆的位置到目标站点j
            for vehicle in vehicle_list:
                if vehicle.no in selected_vehicles:
                    vehicle.current_station = key[1]

    # 计算现在各站点的车辆数
    station_vehicle_count = {}
    for vehicle in vehicle_list:
        station = vehicle.current_station  # 假设车辆对象有 current_station 属性
        station_vehicle_count[station] = station_vehicle_count.get(station, 0) + 1

    # 直接比较（O(S) 时间）
    for i in station_id_list:
        expected = sensor_num_first_day_dic[i]
        actual = station_vehicle_count.get(i, 0)
        if expected != actual:
            print(f"站点 {i} 车辆数不一致：期望 {expected}，实际 {actual}, 第{day}天的再平衡出现问题")

    # 存储下一天的车辆轨迹，更新vehicle_list
    # 补充新的一天的轨迹
    base_dir = "data/数据清洗之后的单车数据/trip_data_march_every_day_6_20_new"
    # 动态构建文件名
    date_str = f'2024-03-{day + 1:02d}'
    filename = f"trip_data_{date_str}.csv"
    file_path = os.path.join(base_dir, filename)
    # 读取这一天的订单数据
    trip_data = pd.read_csv(file_path)
    trip_id_start_node_id_pair_dic = {}  # 存储trip和其对应的开始站点
    trip_id_end_node_id_pair_dic = {}  # 存储trip和其对应的结束站点
    for jjj in range(len(trip_data)):
        trip_id_start_node_id_pair_dic[jjj] = int(trip_data.loc[jjj, "start_node_id"])
        trip_id_end_node_id_pair_dic[jjj] = int(trip_data.loc[jjj, "end_node_id"])
    # 读取第一天的轨迹数据
    with open(f"data/数据清洗之后的单车数据/最短路随机车辆轨迹/" + str(day + 1) + "_route.json", 'r') as f:
        vehicle_route_data = json.load(f)

    empty_route_no_list = []
    for key, value in vehicle_route_data.items():
        if len(value) == 0:
            empty_route_no_list.append(key)

    vehicle_choose_list = []  #存储已经被更新过的车辆编号，避免被重复更新
    for station, sensor_num in sensor_num_first_day_dic.items():
        if int(sensor_num) == 0:
            continue
        else:
            # 读取第day天开始时刻站点station的车辆数
            all_num = int(start_vehicle_num_every_day_dic[(day + 1, station)])
            route_no_list = []
            # 从vehicle_route_data中找到所有从该站点出发的路线编号
            for key, value in vehicle_route_data.items():
                if len(value) != 0:
                    if trip_id_start_node_id_pair_dic[value[0][1]] == station:
                        route_no_list.append(key)
            # 看一下现在找到的这个站点的车辆数
            num = len(route_no_list)
            gap = all_num - num
            if gap != 0:
                # 把empty_route_no_list中前gap辆车的站点改掉
                choose_empty_route_no_list = empty_route_no_list[:gap]
                del empty_route_no_list[:gap]  # 删除
            else:
                choose_empty_route_no_list = []
            all_route_no_list = route_no_list + choose_empty_route_no_list
            if len(all_route_no_list) != all_num:
                print("路线查找存在问题")
            # 从all_route_no_list中随机选择sensor_num条轨迹
            choose_result_list = random.sample(all_route_no_list, int(sensor_num))
            # 根据选出的路线更新车辆轨迹
            # 找到当前站点在station的车辆列表
            station_vehicle_list = find_vehicle_according_station(vehicle_list, station)
            station_vehicle_list = [x for x in station_vehicle_list if x not in vehicle_choose_list]

            if len(station_vehicle_list) < sensor_num:
                raise RuntimeError(f"站点 {station} 车辆不足 {sensor_num} 辆")

            # 按顺序一一对应
            for idx, k in enumerate(choose_result_list):
                vehicle_no = station_vehicle_list[idx]
                vehicle_choose_list.append(vehicle_no)
                for jjj in vehicle_list:
                    if jjj.no == vehicle_no:
                        # 写轨迹
                        jjj.traj[day + 1] = vehicle_route_data.get(str(k), [])  # 空轨迹则用 []
                        # 更新末站
                        if jjj.traj[day + 1]:
                            jjj.current_station = trip_id_end_node_id_pair_dic[vehicle_route_data[str(k)][-1][1]]
                        else:
                            # 空轨迹 → 留在原站
                            pass

# 存储车辆轨迹
save_trajectories(vehicle_list, "data/传感器布设优化/车辆路线每月/" + str(N_k) + "_vehicle_trajectories.json")