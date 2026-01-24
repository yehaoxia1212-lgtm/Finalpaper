# -*- coding: utf-8 -*-
import os
import re
import glob
import random
import numpy as np
import pandas as pd


def fit_k_one_day(
    day_csv: str,
    edge_list: list,
    out_csv: str,
    car_count_initial: int = 300,
    step: int = 10,
    num_simulations: int = 50,
    seed: int = 42,
):
    """
    对单日数据拟合每条 edge 的 k、R^2：
    - 先对 (cab_id, edge_id) 去重 => “一辆车一天内对一条边最多贡献一次覆盖”
    - Monte Carlo 随机抽样不同规模车队，统计覆盖车辆数的期望
    - 过原点最小二乘拟合 y = kx，得到 k ~ p_e
    """
    rng = random.Random(seed)

    data = pd.read_csv(day_csv)
    if "cab_id" not in data.columns or "edge_id" not in data.columns:
        raise ValueError(f"{day_csv} 缺少 cab_id/edge_id 列。当前列：{data.columns.tolist()}")

    # 只保留必要字段，并去重（核心：覆盖车辆数按“不同车辆”计）
    data = data[["cab_id", "edge_id"]].dropna()
    # 强制 edge_id 类型与 edge_list 一致（这里假设 edge_list 是 int）
    data["edge_id"] = pd.to_numeric(data["edge_id"], errors="coerce")
    data = data.dropna(subset=["edge_id"])
    data["edge_id"] = data["edge_id"].astype(int)

    data = data.drop_duplicates(["cab_id", "edge_id"])

    # 每辆车覆盖的边集合（当天）
    car2edges = data.groupby("cab_id")["edge_id"].apply(list).to_dict()
    list_car = list(car2edges.keys())

    if len(list_car) == 0:
        # 当天无数据，直接输出空表
        pd.DataFrame(columns=["edge_id", "k_value", "r_squared"]).to_csv(out_csv, index=False)
        return

    # 当天车数可能不足 300：自动缩小 car_count_initial
    car_count_initial = min(car_count_initial, len(list_car))
    # 保证能被 step 整除，避免最后一个档位奇怪
    car_count_initial = (car_count_initial // step) * step
    if car_count_initial < step:
        pd.DataFrame(columns=["edge_id", "k_value", "r_squared"]).to_csv(out_csv, index=False)
        return

    # 预先建好计数容器：num -> edge -> count
    cover_dic = {
        num: {e: 0 for e in edge_list}
        for num in range(step, car_count_initial + 1, step)
    }

    # Monte Carlo
    for s in range(num_simulations):
        car_left = list_car.copy()
        rng.shuffle(car_left)

        # 你原来是“嵌套抽样”（300,290,...）：这里保持一致
        for car_count in range(car_count_initial, 0, -step):
            selected_cars = car_left[:car_count]
            car_left = selected_cars
            num_now = len(selected_cars)

            # 累计覆盖车辆数：每辆车对每条边最多 +1（因为 car2edges 已经去重）
            for car in selected_cars:
                for e in car2edges.get(car, []):
                    # 防止数据里出现 edge_list 之外的 edge
                    if e in cover_dic[num_now]:
                        cover_dic[num_now][e] += 1

    # 取期望
    avg_cover = {
        num: {e: cover_dic[num][e] / num_simulations for e in edge_list}
        for num in range(step, car_count_initial + 1, step)
    }

    # 拟合：对每条 edge 做 y = kx（过原点）
    k_values = []
    x = np.array([i for i in range(0, car_count_initial + 1, step)], dtype=float)  # 0..N
    denom = np.sum(x ** 2)
    if denom == 0:
        denom = 1.0

    for e in edge_list:
        y_list = [avg_cover[num][e] for num in range(step, car_count_initial + 1, step)]
        if max(y_list) <= 0:
            # 完全没覆盖到就不写（或写 0 都行；这里写 0 更方便后面平均）
            k_values.append({"edge_id": e, "k_value": 0.0, "r_squared": 0.0})
            continue

        y = np.array([0.0] + y_list, dtype=float)  # 加 (0,0)
        k = float(np.sum(x * y) / denom)

        y_pred = k * x
        y_mean = np.mean(y)
        total_var = np.sum((y - y_mean) ** 2)
        resid_var = np.sum((y - y_pred) ** 2)
        r2 = float(1 - (resid_var / total_var)) if total_var != 0 else 0.0

        # 数值保险：k 理论上应在 [0,1]，但受噪声可略超；这里裁剪避免后续分箱问题
        if k < 0:
            k = 0.0
        if k > 1:
            k = 1.0

        k_values.append({"edge_id": e, "k_value": k, "r_squared": r2})

    pd.DataFrame(k_values).to_csv(out_csv, index=False)


if __name__ == "__main__":
    city = "manhattan"

    # ============ 1) 读取 edge_list ============
    edge_data = pd.read_csv(f"./roadmap_taxi_{city}/edge.csv")
    edge_list = pd.to_numeric(edge_data["edge_id"], errors="coerce").dropna().astype(int).unique().tolist()

    # ============ 2) 找到 7 天文件 ============
    data_dir = f"./data_taxi"  # 按你实际目录改
    file_pattern = os.path.join(data_dir, f"data_taxi_{city}_*.csv")  # 例如 data_taxi_manhattan_2024-01-01.csv
    day_files = sorted(glob.glob(file_pattern))

    if len(day_files) == 0:
        raise FileNotFoundError(f"没找到日文件：{file_pattern}")

    out_dir = f"./res_taxi/taxi_k_daily/{city}"
    os.makedirs(out_dir, exist_ok=True)

    # 从文件名提取日期（YYYY-MM-DD）
    date_re = re.compile(r"(\d{4}-\d{2}-\d{2})")

    daily_results = []
    for fp in day_files:
        m = date_re.search(os.path.basename(fp))
        day = m.group(1) if m else os.path.splitext(os.path.basename(fp))[0]

        out_csv = os.path.join(out_dir, f"taxi_k_values_{day}.csv")
        print(f"[DAY] fitting: {day}  file={fp}")

        fit_k_one_day(
            day_csv=fp,
            edge_list=edge_list,
            out_csv=out_csv,
            car_count_initial=300,
            step=10,
            num_simulations=50,
            seed=42,  # 固定种子便于复现；你也可以用 seed=42+idx
        )

        df_day = pd.read_csv(out_csv)
        df_day["date"] = day
        daily_results.append(df_day)

    # ============ 3) 7 天取平均 ============
    all_days = pd.concat(daily_results, ignore_index=True)

    week_mean = (
        all_days
        .groupby("edge_id", as_index=False)
        .agg(
            k_value_mean=("k_value", "mean"),
            r2_mean=("r_squared", "mean"),
            k_value_std=("k_value", "std"),
            n_days=("date", "nunique"),
        )
    )

    week_out = f"./res_taxi/taxi_k_week/{city}"
    os.makedirs(week_out, exist_ok=True)
    week_mean.to_csv(os.path.join(week_out, "taxi_k_values_week_mean.csv"), index=False)

    print("完成：")
    print("  每日 k 输出目录：", out_dir)
    print("  周平均 k：", os.path.join(week_out, "taxi_k_values_week_mean.csv"))
