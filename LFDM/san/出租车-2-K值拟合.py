# -*- coding: utf-8 -*-
import os
import re
import json
import glob
import random
from collections import defaultdict

import numpy as np
import pandas as pd


# =========================================================
# 1) 从 JSONL / per-cab JSON 读取：构建 day -> {cab_id: set(edge_id)}
# =========================================================
def build_daily_car2edges_from_jsonl(jsonl_path: str):
    """
    读取 final_all_cabs.jsonl（每行一个 cab 的输出），返回：
      day2car2edges[date][cab_id] = set(edge_id)
    JSONL 每行格式假设为：
      {"cab_id": "...", "trips":[{"date":"YYYY-MM-DD", "edge_seq":[...]} , ...]}
    """
    day2car2edges = defaultdict(lambda: defaultdict(set))

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                cab_obj = json.loads(line)
            except Exception:
                raise ValueError(f"JSONL 第 {ln} 行解析失败：请检查是否是标准 JSONL（每行一个JSON对象）")

            cab_id = str(cab_obj.get("cab_id", "")).strip()
            if not cab_id:
                continue

            trips = cab_obj.get("trips", []) or []
            for t in trips:
                day = t.get("date", None)
                if not day:
                    continue
                seq = t.get("edge_seq", []) or []
                # seq 里理论上都是 int，但这里做一次保险
                for e in seq:
                    if e is None:
                        continue
                    try:
                        day2car2edges[day][cab_id].add(int(e))
                    except Exception:
                        continue

    return day2car2edges


def build_daily_car2edges_from_cab_json_dir(cab_json_dir: str, pattern="cab_*_trips.json"):
    """
    如果你没有 JSONL，只有每车一个 cab_x_trips.json，也可以用这个。
    """
    day2car2edges = defaultdict(lambda: defaultdict(set))

    files = sorted(glob.glob(os.path.join(cab_json_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"目录下没找到 {pattern}: {cab_json_dir}")

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            cab_obj = json.load(f)

        cab_id = str(cab_obj.get("cab_id", "")).strip()
        if not cab_id:
            continue

        trips = cab_obj.get("trips", []) or []
        for t in trips:
            day = t.get("date", None)
            if not day:
                continue
            seq = t.get("edge_seq", []) or []
            for e in seq:
                if e is None:
                    continue
                try:
                    day2car2edges[day][cab_id].add(int(e))
                except Exception:
                    continue

    return day2car2edges


# =========================================================
# 2) 单日拟合（输入 car2edges：每车当天覆盖的边集合）
# =========================================================
def fit_k_one_day_from_car2edges(
    car2edges: dict,
    edge_list: list,
    out_csv: str,
    car_count_initial: int = 300,
    step: int = 10,
    num_simulations: int = 50,
    seed: int = 42,
):
    """
    输入：
      car2edges: {cab_id: set(edge_id)}  (同一天)
      edge_list: 全路网 edge_id 列表
    输出：
      out_csv: edge_id, k_value, r_squared
    """
    rng = random.Random(seed)

    list_car = list(car2edges.keys())
    if len(list_car) == 0:
        pd.DataFrame(columns=["edge_id", "k_value", "r_squared"]).to_csv(out_csv, index=False)
        return

    # 当天车数可能不足 300：自动缩小 car_count_initial
    car_count_initial = min(car_count_initial, len(list_car))
    car_count_initial = (car_count_initial // step) * step
    if car_count_initial < step:
        pd.DataFrame(columns=["edge_id", "k_value", "r_squared"]).to_csv(out_csv, index=False)
        return

    # 用 defaultdict(int) 只统计出现过的边，避免初始化 num×edge_list 的巨型字典
    cover_dic = {num: defaultdict(int) for num in range(step, car_count_initial + 1, step)}

    # Monte Carlo（保持你原来的“嵌套抽样”逻辑）
    for _ in range(num_simulations):
        car_left = list_car.copy()
        rng.shuffle(car_left)

        for car_count in range(car_count_initial, 0, -step):
            selected_cars = car_left[:car_count]
            car_left = selected_cars
            num_now = len(selected_cars)

            counter = cover_dic[num_now]
            for car in selected_cars:
                for e in car2edges.get(car, []):
                    counter[int(e)] += 1

    # 拟合：y = kx（过原点）
    x = np.array([i for i in range(0, car_count_initial + 1, step)], dtype=float)
    denom = float(np.sum(x ** 2)) if np.sum(x ** 2) != 0 else 1.0

    k_values = []

    # 预取每个num的counter，查不到就当0
    num_list = list(range(step, car_count_initial + 1, step))

    for e in edge_list:
        # y_list: 不同车队规模下“覆盖该边的车辆数期望”
        y_list = []
        for num in num_list:
            y_list.append(cover_dic[num].get(int(e), 0) / num_simulations)

        if max(y_list) <= 0:
            k_values.append({"edge_id": int(e), "k_value": 0.0, "r_squared": 0.0})
            continue

        y = np.array([0.0] + y_list, dtype=float)
        k = float(np.sum(x * y) / denom)

        y_pred = k * x
        y_mean = np.mean(y)
        total_var = float(np.sum((y - y_mean) ** 2))
        resid_var = float(np.sum((y - y_pred) ** 2))
        r2 = float(1 - (resid_var / total_var)) if total_var != 0 else 0.0

        # 裁剪避免后续分箱怪异
        if k < 0:
            k = 0.0
        if k > 1:
            k = 1.0

        k_values.append({"edge_id": int(e), "k_value": k, "r_squared": r2})

    pd.DataFrame(k_values).to_csv(out_csv, index=False)


# =========================================================
# 3) 主程序：旧金山（从 JSONL 读每一天 car2edges，然后按天拟合，最后周平均）
# =========================================================
if __name__ == "__main__":
    city = "san"

    # --------- A) edge_list 读取（你可选 edge.csv 或 edge.geojson 导出的 csv）---------
    # 你之前曼哈顿用 edge.csv；旧金山如果没有 edge.csv，你也可以把 edge.geojson 先导出一个 edge.csv
    EDGE_CSV = f"./roadmap_taxi_{city}/edge.csv"  # 确保有 edge_id 列
    edge_data = pd.read_csv(EDGE_CSV)
    edge_list = pd.to_numeric(edge_data["edge_id"], errors="coerce").dropna().astype(int).unique().tolist()

    # --------- B) 输入：你的最终轨迹结果（推荐 JSONL）---------
    # 这个就是你“补全+去刺头”之后的总文件：final_all_cabs.jsonl
    FINAL_JSONL = f"./data_taxi/final_clean_trips/final_all_cabs.jsonl"  # 按你实际路径改
    # 如果你不是 jsonl，而是每车一个cab_x_trips.json，改用 build_daily_car2edges_from_cab_json_dir

    day2car2edges = build_daily_car2edges_from_jsonl(FINAL_JSONL)
    all_days = sorted(day2car2edges.keys())
    if not all_days:
        raise ValueError("从 JSONL 没读到任何 date/trips，请检查文件内容是否为最终 trips 输出。")

    out_dir = f"./res_taxi/taxi_k_daily/{city}"
    os.makedirs(out_dir, exist_ok=True)

    daily_results = []
    for day in all_days:
        car2edges = day2car2edges[day]
        out_csv = os.path.join(out_dir, f"taxi_k_values_{day}.csv")
        print(f"[DAY] {day}  cars={len(car2edges)}  -> {out_csv}")

        fit_k_one_day_from_car2edges(
            car2edges=car2edges,
            edge_list=edge_list,
            out_csv=out_csv,
            car_count_initial=300,
            step=10,
            num_simulations=50,
            seed=42,
        )

        df_day = pd.read_csv(out_csv)
        df_day["date"] = day
        daily_results.append(df_day)

    # --------- C) 周平均（按你原来的方式）---------
    all_days_df = pd.concat(daily_results, ignore_index=True)

    week_mean = (
        all_days_df
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
    week_path = os.path.join(week_out, "taxi_k_values_week_mean.csv")
    week_mean.to_csv(week_path, index=False)

    print("完成：")
    print("  每日 k 输出目录：", out_dir)
    print("  周平均 k：", week_path)
