import pandas as pd
import os
from gurobipy import *
import pickle
from math import floor
import numpy as np
from collections import defaultdict

# 读取所有的站点数据
station_data = pd.read_csv("data/数据清洗之后的单车数据/每天开始时刻各站点所需的车辆数.csv")
station_id_list = station_data["station_id"].tolist()
station_id_list = [int(i) for i in station_id_list]

file_route = "data/Manhattan_bike_road_network/"
# 读取邻接矩阵数据（numpy格式）
with open(file_route + "shortest_lengths_dict.pickle", 'rb') as f:
    shortest_lengths_dict = pickle.load(f)

# 日期列表
total_days = 31
all_days = list(range(1, total_days + 1))        # 1-31日
days_from_2 = list(range(2, total_days + 1))     # 2-31日
dispatch_days = list(range(1, total_days))       # 1-30日（调度日）

# 存储第t天结束时刻相较于开始时刻各站点的车辆变化值,格式（日期，站点）：变化值
delta_fleet_size_every_day_dic = {}
# 遍历日期
for no in range(1, 32):
    # 动态构建文件名
    date_str = f'2024-03-{no:02d}'  # 格式化为YYYY-MM-DD
    if no < 10:
        filename = f"trip_data_{date_str}.csv"
    else:
        filename = f"trip_data_{date_str}.csv"
    file_path = os.path.join("data/数据清洗之后的单车数据/vehicle_num_new", filename)
    # 读取CSV文件
    df = pd.read_csv(file_path, usecols=['station_id', '1200'])
    # 遍历每一行数据
    for _, row in df.iterrows():
        station_id = int(row['station_id'])
        fleet_size = row['1200']
        # 将站点需求添加到当天的字典中
        delta_fleet_size_every_day_dic[(no, station_id)] = fleet_size

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


# 读取共享单车调度数据（字典格式）
with open("data/数据清洗之后的单车数据/仅考虑运营情况下共享单车每天的调度计划.pickle", 'rb') as f:
    bike_dispatch_dic = pickle.load(f)

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


def Model_first_day(station_id_list, edge_list, edge_length_dic, start_vehicle_num_every_day_dic, k_value_dic):
    K = 1
    M = 1000  # 一个很大的数

    # 创建一个模型
    D1_model = Model("D1_model")

    # 定义决策变量
    # 每天开始时刻各站点的传感器数量
    x_start_1 = D1_model.addVars(station_id_list, lb=0, vtype=GRB.INTEGER, name="x_start_1")

    # y_ew变量，表示时空路段e的覆盖是否满足需求
    y_e_1 = D1_model.addVars(edge_list, vtype=GRB.BINARY, name="y_ew")

    # N_ew表示路段e的期望覆盖次数
    N_e_1 = D1_model.addVars(edge_list, vtype=GRB.CONTINUOUS, name="N_ew")

    # 添加优化目标(最小化调度距离)
    D1_model.setObjective(
        quicksum(edge_length_dic[e] * y_e_1[e] for e in edge_list),
        GRB.MAXIMIZE
    )
    # 添加约束条件

    # 传感器数量约束
    for s in station_id_list:
        D1_model.addConstr((x_start_1[s] <= start_vehicle_num_every_day_dic[(4, s)]), name="eq_1")

    D1_model.addConstr((quicksum(x_start_1[s] for s in station_id_list) == N_k), name="eq_2")

    # 覆盖相关约束
    for e in edge_list:
        D1_model.addConstr((N_e_1[e] == quicksum(k_value_dic[(s, e)] * x_start_1[s] for s in station_id_list)), name="eq_3_23")
        D1_model.addConstr(-M * (1 - y_e_1[e]) <= N_e_1[e] - K, name=f"cover_low_{e}")
        D1_model.addConstr((N_e_1[e] - K <= M * y_e_1[e]), name="eq_3_24")

    # 设置求解时间限制(3600秒)
    D1_model.setParam(GRB.Param.TimeLimit, 3600)

    # 7. 优化求解与诊断
    D1_model.Params.LogToConsole = 1  # 启用求解日志
    D1_model.optimize()

    # 结果处理
    if D1_model.status == GRB.OPTIMAL:
        print("\n优化成功!")
        print(f"优化目标为: {D1_model.ObjVal:.2f}")

    # 记录第一天各站点分配的传感器数量
    # 输出每个站点布设的传感器数量
    sensor_num_dic = {}
    for s in station_id_list:
        sensor_num_dic[s] = x_start_1[s].x
    return sensor_num_dic

N_k = 100
first_day_sensor_num_dic = Model_first_day(station_id_list, edge_list, edge_length_dic, start_vehicle_num_every_day_dic, k_value_dic)

# 存储数据

# sensor_num_dic 形如 {station_id: 整数}
df = pd.DataFrame(
    {"Sensor": first_day_sensor_num_dic.keys(),
     "Count":      first_day_sensor_num_dic.values()}
)
df.to_csv("data/传感器布设优化/Four_day/sensor_first_day_" + str(N_k) + ".csv", index=False, columns=["Sensor", "Count"])
