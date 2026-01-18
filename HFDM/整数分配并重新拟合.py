import os
import ast
import json
import numpy as np
import pandas as pd
import geopandas as gpd


# =========================
# 0) 通用：保存每个时间片的热力图 Shapefile
# =========================
def save_heat_shp(
    city: str,
    resolution: int,
    scheme: str,            # "taxi" / "bus" / "joint"
    num_sensors: int,
    Net: np.ndarray,        # shape: (num_edge, num_time)
    base_gdf: gpd.GeoDataFrame,  # 只含 grid_id/geometry，且顺序与 Net 的 edge 维度一致
    time_key: list,
    beta: float = 0.2,
    save_power: bool = True,     # True: 保存 Net**beta；False: 保存 Net
    root_dir: str = "./res_heat"
):
    """
    输出结构：
      ./res_heat/{scheme}/{city}/grid{resolution}/num_{num_sensors}/time_{time}.shp

    输出字段（三列）：
      grid_id, geometry, score
    """
    out_dir = os.path.join(root_dir, scheme)
    os.makedirs(out_dir, exist_ok=True)

    gdf_out = base_gdf.copy()  # grid_id/geometry

    for t_idx, t_val in enumerate(time_key):
        vec = Net[:, t_idx]  # (num_edge,)
        score_vec = (vec ** beta) if save_power else vec
        gdf_out["score"] = score_vec.astype(float)

        shp_path = os.path.join(out_dir, f"{city}_{num_sensors}_grid{resolution}_time{int(t_val)}.shp")
        gdf_out.to_file(shp_path, driver="ESRI Shapefile", encoding="utf-8")
        csv_path = os.path.join(out_dir, f"{city}_{num_sensors}_grid{resolution}_time{int(t_val)}.csv")
        gdf_out.to_csv(csv_path, index=False)


# =========================
# 1) 三种方案：joint / bus / taxi（计算 score + 可选保存热力图 shp）
# =========================
def taxi_bus(
    city, resolution,
    num_edge, num_time, time_key,
    matrix_taxi, matrix_bus, service_intensity, time_total, num_bus_total,
    base_gdf,
    beta=0.2, save_heat=True, save_power=True
):
    out = []
    in_path = f'./res_joint/{city}/opvalue_vs_numbus-taxi_grid{resolution}.csv'
    number_grids = pd.read_csv(in_path)

    os.makedirs(f'./res_joint/{city}', exist_ok=True)

    for _, row in number_grids.iterrows():
        num_sensor = int(row['num_sensors'])

        # ---------- 1) taxi/bus 总量整数化：最大余数法 ----------
        taxi_int = int(row['taxi_number'])
        bus_int = int(num_sensor - row['taxi_number'])

        current_sum = taxi_int + bus_int
        remaining = num_sensor - current_sum

        if remaining > 0:
            fractions = [
                ('taxi', row['taxi_number'] - taxi_int),
                ('bus', (num_sensor - row['taxi_number']) - bus_int)
            ]
            for name_, _ in sorted(fractions, key=lambda x: -x[1]):
                if remaining <= 0:
                    break
                if name_ == 'taxi':
                    taxi_int += 1
                else:
                    bus_int += 1
                remaining -= 1

        num_taxi = taxi_int
        num_bus = bus_int

        # ---------- 2) 公交线路层整数化：最大余数法 ----------
        bus_list = ast.literal_eval(row['bus_sensors_list'])
        int_bus_list = [int(x) for x in bus_list]
        current_sum_bus = sum(int_bus_list)
        remaining_bus = num_bus - current_sum_bus

        if remaining_bus > 0:
            bus_fracs = [(i, x - int(x)) for i, x in enumerate(bus_list)]
            for i, _ in sorted(bus_fracs, key=lambda x: -x[1]):
                if remaining_bus <= 0:
                    break
                int_bus_list[i] += 1
                remaining_bus -= 1

        bus_list_final = int_bus_list

        if len(bus_list_final) != num_bus_total:
            raise ValueError(
                f"[JOINT] bus_sensors_list length ({len(bus_list_final)}) != num_bus_total ({num_bus_total})."
            )

        # ---------- 3) 计算 Net[e,t] ----------
        Net = np.zeros((num_edge, num_time), dtype=float)

        for t in range(num_time):
            key_time = str(time_key[t])
            p_vec = matrix_taxi[key_time]  # (num_edge,)

            # taxi 项
            Net[:, t] = num_taxi * p_vec

            # bus 项叠加
            # matrix_bus: (num_bus_total, num_edge)
            # service_intensity: (num_bus_total, num_time)
            # time_total: list length num_bus_total
            for j in range(num_bus_total):
                if bus_list_final[j] == 0:
                    continue
                Net[:, t] += (
                    bus_list_final[j]
                    * matrix_bus[j, :]
                    * service_intensity[j][t]
                    * 60.0 / time_total[j]
                )

        score = float(np.sum(Net ** beta))
        print(f"[JOINT] city={city}, grid={resolution}, num={num_sensor}, score={score}")

        out.append({
            'num_sensors': num_sensor,
            'score': score,
            'taxi_number': num_taxi,
            'bus_number': num_bus,
            'bus_sensors_list': bus_list_final,
        })

        if save_heat:
            save_heat_shp(
                city=city, resolution=resolution, scheme="joint",
                num_sensors=num_sensor, Net=Net, base_gdf=base_gdf, time_key=time_key,
                beta=beta, save_power=save_power, root_dir="./res_heat"
            )

    pd.DataFrame(out).to_csv(f'./res_joint/{city}/Score-bus-taxi_grid{resolution}.csv', index=False)


def bus_only(
    city, resolution,
    num_edge, num_time, time_key,
    matrix_bus, service_intensity, time_total, num_bus_total,
    base_gdf,
    beta=0.2, save_heat=True, save_power=True
):
    out = []
    in_path = f'./res_bus/{city}/opvalue_vs_numbus_grid{resolution}.csv'
    number_grids = pd.read_csv(in_path)

    os.makedirs(f'./res_bus/{city}', exist_ok=True)

    for _, row in number_grids.iterrows():
        num_sensor = int(row['num_sensors'])

        bus_list = ast.literal_eval(row['bus_sensors_list'])
        int_bus_list = [int(x) for x in bus_list]
        current_sum_bus = sum(int_bus_list)
        remaining_bus = num_sensor - current_sum_bus

        if remaining_bus > 0:
            bus_fracs = [(i, x - int(x)) for i, x in enumerate(bus_list)]
            for i, _ in sorted(bus_fracs, key=lambda x: -x[1]):
                if remaining_bus <= 0:
                    break
                int_bus_list[i] += 1
                remaining_bus -= 1

        bus_list_final = int_bus_list

        if len(bus_list_final) != num_bus_total:
            raise ValueError(
                f"[BUS] bus_sensors_list length ({len(bus_list_final)}) != num_bus_total ({num_bus_total})."
            )

        Net = np.zeros((num_edge, num_time), dtype=float)

        for t in range(num_time):
            for j in range(num_bus_total):
                if bus_list_final[j] == 0:
                    continue
                Net[:, t] += (
                    bus_list_final[j]
                    * matrix_bus[j, :]
                    * service_intensity[j][t]
                    * 60.0 / time_total[j]
                )

        score = float(np.sum(Net ** beta))
        print(f"[BUS] city={city}, grid={resolution}, num={num_sensor}, score={score}")

        out.append({
            'num_sensors': num_sensor,
            'score': score,
            'bus_sensors_list': bus_list_final,
        })

        if save_heat:
            save_heat_shp(
                city=city, resolution=resolution, scheme="bus",
                num_sensors=num_sensor, Net=Net, base_gdf=base_gdf, time_key=time_key,
                beta=beta, save_power=save_power, root_dir="./res_heat"
            )

    pd.DataFrame(out).to_csv(f'./res_bus/{city}/Score-bus_grid{resolution}.csv', index=False)


def taxi_only(
    city, resolution,
    num_edge, num_time, time_key,
    matrix_taxi,
    base_gdf,
    beta=0.2, num_sensors_list=None, save_heat=True, save_power=True
):
    out = []
    os.makedirs(f'./res_taxi/{city}', exist_ok=True)

    # 默认用 joint 文件的 num_sensors 作为横轴，保证三条曲线点一致
    if num_sensors_list is None:
        joint_path = f'./res_joint/{city}/opvalue_vs_numbus-taxi_grid{resolution}.csv'
        df = pd.read_csv(joint_path)
        num_sensors_list = df['num_sensors'].astype(int).tolist()

    for num_sensor in num_sensors_list:
        n_T = int(num_sensor)
        Net = np.zeros((num_edge, num_time), dtype=float)

        for t in range(num_time):
            key_time = str(time_key[t])
            p_vec = matrix_taxi[key_time]   # (num_edge,)
            Net[:, t] = n_T * p_vec         # N^T_{g,t}

        score = float(np.sum(Net ** beta))
        print(f"[TAXI] city={city}, grid={resolution}, num={n_T}, score={score}")

        out.append({
            'num_sensors': n_T,
            'score': score,
            'taxi_number': n_T
        })

        if save_heat:
            save_heat_shp(
                city=city, resolution=resolution, scheme="taxi",
                num_sensors=n_T, Net=Net, base_gdf=base_gdf, time_key=time_key,
                beta=beta, save_power=save_power, root_dir="./res_heat"
            )

    pd.DataFrame(out).to_csv(f'./res_taxi/{city}/Score-taxi_grid{resolution}.csv', index=False)


# =========================
# 2) 主程序：循环 city / resolution，构建矩阵并输出三类结果 + 热力图 shp
# =========================
if __name__ == "__main__":

    cities = ['manhattan', 'chengdu', 'san']
    ress = [1000, 500]

    # 时间：6:00 到 20:00（不含 20:00），每小时一个时间片
    # time_key: [360, 420, ..., 1140]
    time_key = list(range(360, 1200, 60))
    num_time = len(time_key)

    beta = 0.2
    save_heat = True          # 是否输出 shp
    save_power = True         # True 保存 Net**beta（每网格效用贡献）；False 保存 Net（覆盖强度）

    for city in cities:
        for resolution in ress:
            print(f"\n========== Running: city={city}, grid={resolution} ==========")

            # ---------- 1) 读取网格（必须保持稳定顺序，不能 set()） ----------
            grid_path = f'./data_grid/{city}_grid/{city}_grid{resolution}.geojson'
            data_grid = gpd.read_file(grid_path)
            data_grid = data_grid[['grid_id', 'geometry']].drop_duplicates(subset=['grid_id']).reset_index(drop=True)

            grid_ids = data_grid['grid_id'].tolist()
            grid2idx = {gid: i for i, gid in enumerate(grid_ids)}
            num_edge = len(grid_ids)

            # base_gdf：顺序与 Net 的 edge 维度一致（非常关键）
            base_gdf = data_grid[['grid_id', 'geometry']].copy()

            # ---------- 2) 出租车：构建 matrix_taxi（key=str(time), val=vector p_{g,t}） ----------
            taxi_k_path = f'./res_taxi/taxi_k/{city}/taxi_k_values_grid{resolution}.csv'
            data_taxi = pd.read_csv(taxi_k_path)
            data_taxi["time_key"] = data_taxi["time_key"].astype(int)
            data_taxi["grid_id"] = data_taxi["grid_id"].astype(int)

            taxi_mat = np.zeros((num_time, num_edge), dtype=float)
            time2idx = {int(t): i for i, t in enumerate(time_key)}

            for _, r in data_taxi.iterrows():
                tk = int(r["time_key"])
                gi = int(r["grid_id"])
                kv = float(r["k_value"])
                ti = time2idx.get(tk, None)
                ei = grid2idx.get(gi, None)
                if ti is not None and ei is not None:
                    taxi_mat[ti, ei] = kv

            matrix_taxi = {str(time_key[t]): taxi_mat[t, :] for t in range(num_time)}

            # ---------- 3) 公交：构建 matrix_bus（线路×网格），service_intensity（线路×时间） ----------
            bus_path = f"./data_bus/{city}/data_bus_{city}_grid{resolution}.csv"
            data_bus = pd.read_csv(bus_path)
            data_bus["pass_grid_id"] = data_bus["pass_grid_id"].apply(lambda s: json.loads(s) if pd.notna(s) else [])

            bus_id = data_bus["bus_id"].tolist()
            time_total = data_bus["time_total"].tolist()  # 确保单位与 60/time_total 一致（分钟推荐）
            num_bus_total = len(bus_id)

            # service_intensity：这里读一个通用曲线并复制到每条线路（你原逻辑）
            service_intensity_data = pd.read_csv("./data_bus/bus_service_intensity.csv")
            intensity_list = service_intensity_data["intensity"].values.tolist()
            if len(intensity_list) != num_time:
                raise ValueError(
                    f"service_intensity length ({len(intensity_list)}) != num_time ({num_time})."
                )
            service_intensity = [intensity_list for _ in range(num_bus_total)]  # (num_bus_total, num_time)

            matrix_bus = np.zeros((num_bus_total, num_edge), dtype=int)
            for i, r in data_bus.iterrows():
                for gid in r["pass_grid_id"]:
                    idx = grid2idx.get(gid, None)
                    if idx is not None:
                        matrix_bus[i, idx] = 1

            # ---------- 4) 计算三类方案 + 保存热力图 shp ----------
            taxi_bus(
                city=city, resolution=resolution,
                num_edge=num_edge, num_time=num_time, time_key=time_key,
                matrix_taxi=matrix_taxi,
                matrix_bus=matrix_bus,
                service_intensity=service_intensity,
                time_total=time_total,
                num_bus_total=num_bus_total,
                base_gdf=base_gdf,
                beta=beta, save_heat=save_heat, save_power=save_power
            )

            bus_only(
                city=city, resolution=resolution,
                num_edge=num_edge, num_time=num_time, time_key=time_key,
                matrix_bus=matrix_bus,
                service_intensity=service_intensity,
                time_total=time_total,
                num_bus_total=num_bus_total,
                base_gdf=base_gdf,
                beta=beta, save_heat=save_heat, save_power=save_power
            )

            taxi_only(
                city=city, resolution=resolution,
                num_edge=num_edge, num_time=num_time, time_key=time_key,
                matrix_taxi=matrix_taxi,
                base_gdf=base_gdf,
                beta=beta, num_sensors_list=None, save_heat=save_heat, save_power=save_power
            )

    print("\nAll done. Heat shapefiles are saved under ./res_heat/")
