import os
import json
import math
import pickle
import random
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt

from gurobipy import Model, GRB, quicksum

warnings.filterwarnings("ignore")


def network(node_id, node_id_x, node_id_y, length, edge_id):
    """Build an undirected weighted graph."""
    G = nx.Graph()
    G.add_nodes_from(node_id)
    for i in range(len(node_id_x)):
        G.add_edge(node_id_x[i], node_id_y[i], weight=length[i], edge_id=edge_id[i])
    return G


if __name__ == "__main__":
    city = "manhattan"
    resolution = 1000

    # --------------------------
    # 1) Load grid and build mapping
    # --------------------------
    data_grid = gpd.read_file(f"./data_grid/{city}_grid/{city}_grid{resolution}.geojson")

    # 注意：用 set 会打乱顺序，后续索引会不稳定；这里按原表顺序取
    grid_list = data_grid["grid_id"].tolist()
    num_edge = len(grid_list)

    grid2idx = {gid: i for i, gid in enumerate(grid_list)}  # O(1) lookup

    # --------------------------
    # 2) Load bus data
    # --------------------------
    data = pd.read_csv(f"./data_bus/{city}/data_bus_{city}_grid{resolution}.csv")

    # pass_grid_id: JSON string -> list
    data["pass_grid_id"] = data["pass_grid_id"].apply(lambda s: json.loads(s) if pd.notna(s) else [])

    # 时间片（小时级，单位：分钟）
    time_key = [t for t in range(360, 1200, 60)]
    num_time = len(time_key)

    bus_id = data["bus_id"].tolist()
    time_total = data["time_total"].tolist()  # 每条线路（或一次行程）的总运行时长（分钟或秒？需与你 60/ time_total 一致）
    num_bus = len(bus_id)

    # --------------------------
    # 3) Load service intensity (per time slice)
    # --------------------------
    service_intensity_data = pd.read_csv("./data_bus/bus_service_intensity.csv")
    service_intensity_list = service_intensity_data["intensity"].values.tolist()

    if len(service_intensity_list) != num_time:
        raise ValueError(
            f"service_intensity_list length ({len(service_intensity_list)}) "
            f"must equal num_time ({num_time}). "
            f"Check ./data_bus/bus_service_intensity.csv"
        )

    # 你原来写的是“每辆车都用同一条平均强度曲线”
    service_intensity = [service_intensity_list for _ in range(num_bus)]

    # --------------------------
    # 4) Build edge_matrix (bus x grid)
    # --------------------------
    edge_matrix = np.zeros((num_bus, num_edge), dtype=int)

    for i, row in data.iterrows():
        for gid in row["pass_grid_id"]:
            # 如果 pass_grid_id 里出现了不在 grid_list 的 gid，跳过或报错
            idx = grid2idx.get(gid, None)
            if idx is not None:
                edge_matrix[i, idx] = 1

    # --------------------------
    # 5) Optimization loop
    # --------------------------
    out_dir = f"./res_bus/{city}"
    os.makedirs(out_dir, exist_ok=True)

    grids_vs_bus = []
    num_sensors_range = range(10, 301, 10)

    for num_sensor in num_sensors_range:
        model = Model("model_bus")
        model.setParam("MIPGap", 0.03)
        model.setParam("OutputFlag", 1)

        # 决策变量
        # N_et: 每个网格 e 在时间 t 的“平均被覆盖车辆数/强度”（你的定义）
        N_et = model.addVars(num_edge, num_time, vtype=GRB.CONTINUOUS, lb=0.0, name="N_et")

        # y_jB: 分配给第 j 条线路（或行程）的传感器数量（你用连续变量；若必须整数，可改 GRB.INTEGER）
        y_jB = model.addVars(num_bus, vtype=GRB.CONTINUOUS, lb=0.0, ub=num_sensor, name="y_jB")

        # z_et: N_et 的幂次效用变量
        z_et = model.addVars(num_edge, num_time, vtype=GRB.CONTINUOUS, lb=0.0, name="z_et")

        # 幂约束：z = N^0.2
        for e in range(num_edge):
            for t in range(num_time):
                model.addGenConstrPow(N_et[e, t], z_et[e, t], 0.2, name=f"con_pow_{e}_{t}")

        # 目标：最大化总效用
        model.setObjective(
            quicksum(z_et[e, t] for e in range(num_edge) for t in range(num_time)),
            GRB.MAXIMIZE,
        )

        # 约束1：总传感器数
        model.addConstr(quicksum(y_jB[j] for j in range(num_bus)) <= num_sensor, name="con_total_sensors")

        # 约束2：覆盖强度计算（保持你原式不变）
        # N_et[e,t] = Σ_j y_jB[j] * edge_matrix[j,e] * intensity[j,t] * 60 / time_total[j]
        for t in range(num_time):
            for e in range(num_edge):
                model.addConstr(
                    N_et[e, t]
                    == quicksum(
                        y_jB[j]
                        * edge_matrix[j, e]
                        * service_intensity[j][t]
                        * 60.0
                        / time_total[j]
                        for j in range(num_bus)
                    ),
                    name=f"con_cover_e{e}_t{t}",
                )

        model.optimize()

        if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            print(f"[WARN] num_sensor={num_sensor}, model status={model.Status}, skip saving.")
            continue

        print("覆盖到的时空网格数量（目标值）：", model.objVal)

        # 保存每条线路分配的传感器
        bus_choice = [y_jB[i].X for i in range(num_bus)]
        grids_vs_bus.append(
            {"num_sensors": num_sensor, "grids_covered": float(model.objVal), "bus_sensors_list": bus_choice}
        )

        # 保存每个 (e,t) 的 N_et 与效用
        covered_info = []
        for t in range(num_time):
            for e in range(num_edge):
                covered_info.append(
                    {
                        "time_key": time_key[t],
                        "grid_id": grid_list[e],
                        "N_et": float(N_et[e, t].X),
                        "N_et^0.2": float(z_et[e, t].X),
                    }
                )

        covered_df = pd.DataFrame(covered_info)
        covered_df.to_csv(f"{out_dir}/covered_info_sensors{num_sensor}_grid{resolution}.csv", index=False)

    grids_vs_bus_df = pd.DataFrame(grids_vs_bus)
    grids_vs_bus_df.to_csv(f"{out_dir}/opvalue_vs_numbus_grid{resolution}.csv", index=False)
    print("Done.")
