import pandas as pd
import networkx as nx
import random
import datetime
import pickle
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import numpy as np
import ast
import json
# from gurobipy import *
import geopandas as gpd

plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


def network(node_id, node_id_x, node_id_y, length, edge_id):
    G = nx.Graph()  # 创建一个空图
    G.add_nodes_from(node_id)  # 添加点
    for i in range(len(node_id_x)):
        G.add_edge(node_id_x[i], node_id_y[i], weight=length[i], edge_id=edge_id[i])
    return G

if __name__ == '__main__':
    city = 'manhattan'
    resolution = 1000

    out_dir = f"./res_joint/{city}"
    os.makedirs(out_dir, exist_ok=True)

    time_key = [t for t in range(360, 1200, 60)]
    num_time = len(time_key)

    data_grid = gpd.read_file(f"./data_grid/{city}_grid/{city}_grid{resolution}.geojson")
    grid_list = data_grid["grid_id"].tolist()
    num_edge = len(grid_list)
    grid2idx = {gid: i for i, gid in enumerate(grid_list)}  # O(1) lookup

    data_bus = pd.read_csv(f"./data_bus/{city}/data_bus_{city}_grid{resolution}.csv")
    data_bus["pass_grid_id"] = data_bus["pass_grid_id"].apply(lambda s: json.loads(s) if pd.notna(s) else [])
    bus_id = data_bus["bus_id"].tolist()
    time_total = data_bus["time_total"].tolist()  # 每条线路（或一次行程）的总运行时长（分钟或秒？需与你 60/ time_total 一致）
    num_bus = len(bus_id)
    service_intensity_data = pd.read_csv("./data_bus/bus_service_intensity.csv")
    service_intensity_list = service_intensity_data["intensity"].values.tolist()

    if len(service_intensity_list) != num_time:
        raise ValueError(
            f"service_intensity_list length ({len(service_intensity_list)}) "
            f"must equal num_time ({num_time}). "
            f"Check ./data_bus/bus_service_intensity.csv"
        )
    service_intensity = [service_intensity_list for _ in range(num_bus)]
    matrix_bus = np.zeros((num_bus, num_edge), dtype=int)

    for i, row in data_bus.iterrows():
        for gid in row["pass_grid_id"]:
            # 如果 pass_grid_id 里出现了不在 grid_list 的 gid，跳过或报错
            idx = grid2idx.get(gid, None)
            if idx is not None:
                matrix_bus[i, idx] = 1

    #出租车矩阵
    data_taxi = pd.read_csv(f'./res_taxi/taxi_k/{city}/taxi_k_values_grid{resolution}.csv')

    # 保证类型一致
    data_taxi["time_key"] = data_taxi["time_key"].astype(int)
    data_taxi["grid_id"] = data_taxi["grid_id"].astype(int)

    taxi_mat = np.zeros((num_time, num_edge), dtype=float)
    time2idx = {int(t): i for i, t in enumerate(time_key)}

    # 填充：把每条记录写入对应位置
    for _, r in data_taxi.iterrows():
        tk = int(r["time_key"])
        gi = int(r["grid_id"])
        kv = float(r["k_value"])
        ti = time2idx.get(tk, None)
        ei = grid2idx.get(gi, None)  # 你前面 grid2idx 是用 grid_list 建的
        if ti is not None and ei is not None:
            taxi_mat[ti, ei] = kv

    # 转成你需要的 dict：key 用 str(time_key[t])，value 是一维数组 taxi_mat[t, :]
    matrix_taxi = {str(time_key[t]): taxi_mat[t, :] for t in range(num_time)}



    num_sensors = range(10,201,10) # 所拥有的传感器的数目
    grids_vs_bus_and_taxi =[]
    for num_sensor in num_sensors:
        model = Model("model_bus")
        # 定义决策变量
        # y_jB表示j线传感器个数
        y_jB = model.addVars(num_bus,vtype=GRB.CONTINUOUS,lb=0,ub=num_sensor,name="y_jB")

        n_taxi = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=num_sensor, name="n_taxi")

        N_et = model.addVars(num_edge,num_time,vtype=GRB.CONTINUOUS, name="N_e")
        z_et = model.addVars(num_edge, num_time, vtype=GRB.CONTINUOUS, name="z_et", lb=0)

        # 使用 addGenConstrPow 来处理 N_et^0.2 的约束
        for e in range(num_edge):
            for t in range(num_time):
                # 生成约束 z_et[e,t] = N_et[e,t]^0.2
                model.addGenConstrPow(N_et[e, t], z_et[e, t], 0.2, name=f"con_pow_{e}_{t}")

        # 目标函数：最大化 z_et
        model.setObjective(
            quicksum(z_et[e, t] for e in range(num_edge) for t in range(num_time)),
            GRB.MAXIMIZE
        )
        # 添加约束条件
        # 总传感器的数量限制
        model.addConstr((quicksum(y_jB[j] for j in range(num_bus)) + n_taxi <= num_sensor), name='con_1')


        for t in range(num_time):
            key_time = str(time_key[t])
            delta_time_taxi = matrix_taxi[key_time]
            for e in range(num_edge):

                model.addConstr((N_et[e,t] == n_taxi*delta_time_taxi[e]+quicksum(y_jB[j] * matrix_bus[j,e]*service_intensity[j][t]*60/time_total[j] for j in range(num_bus))), name=f'con_2_{num_sensor}_{t}_{e}')


        # 求解
        model.optimize()

        print("覆盖到的时空路段数量：", model.objVal)
        bus_choice = []
        for i in range(num_bus):
            bus_choice.append(y_jB[i].X)

        grids_vs_bus_and_taxi.append({'num_sensors': num_sensor, 'score': model.objVal, 'bus_sensors_list': bus_choice,'taxi_number':n_taxi.X})
        print(grids_vs_bus_and_taxi)
        covered_info = []
        for t in range(num_time):
            for e in range(num_edge):
                time = time_key[t]
                edges = grid_list[e]
                N= N_et[e, t].X
                z = z_et[e, t].X
                covered_info.append({'time_key': time, 'grid_id': edges,'N_et': N, 'N_et^0.2': z})
        covered_df = pd.DataFrame(covered_info)

        covered_df.to_csv(f"{out_dir}/covered_info_sensors{num_sensor}_grid{resolution}.csv", index=False)
    grids_vs_bus_and_taxi_df=pd.DataFrame(grids_vs_bus_and_taxi)
    grids_vs_bus_and_taxi_df.to_csv(f"{out_dir}/opvalue_vs_numbus-taxi_grid{resolution}.csv", index=False)
    print()


