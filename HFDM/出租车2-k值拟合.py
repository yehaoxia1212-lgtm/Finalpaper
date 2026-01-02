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
import geopandas as gpd
import tqdm

if __name__ == '__main__':

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    cities = ['san','manhattan','chengdu']
    resolution = [1000,500]

    for city in cities:
        for res in resolution:
            print(city,res)
            # 构建路网
            data_grid = gpd.read_file(f'./data_grid/{city}_grid/{city}_grid{res}.geojson')
            grid_list = list(set(data_grid['grid_id']))

            data = gpd.read_file(f'./data_taxi/{city}/data_taxi_{city}_grid{res}.geojson')
            data['date_time'] = pd.to_datetime(data['date_time'])
            data['time_minute'] = data['date_time'].dt.hour * 60 + data['date_time'].dt.minute
            data['Time_key'] = (data['time_minute']// 60) * 60
            time_key = [time for time in range(360, 1200, 60)]
            # 获取所有车辆编号
            list_car = list(set(data['cab_id'].values))
            car_count_initial = 300  # 初始车辆数量
            step = 10  # 每次减少的车辆数量
            num_simulations = 50

            cover_dic_all_list_all_time = {
                num: {time: {key: 0 for key in grid_list} for time in range(360, 1200, 60)} for num in
                range(10, car_count_initial + 1, 10)
            }

            for simulation in tqdm.tqdm(range(50)):

                car_left = list_car.copy()  # 复制初始车辆列表
                random.shuffle(car_left)  # 打乱车辆顺序

                for car_count in range(car_count_initial, 0, -step):
                    selected_cars = car_left[:car_count]
                    selected_data = data[data['cab_id'].isin(selected_cars)]  # 筛选数据
                    car_left = selected_cars
                    num_now = len(selected_cars)

                    for car in selected_cars:
                        data_car = selected_data[selected_data['cab_id'] == car]

                        for time in time_key:
                            data_time = data_car[data_car['Time_key'] == time]

                            # ==========================
                            # 必要改动 1：真正“覆盖车辆数”去重
                            # 每辆车在该小时内对同一网格最多贡献一次
                            # ==========================
                            if data_time.empty:
                                continue
                            unique_grids = pd.unique(data_time['grid_id'].values)

                            # （保留你的写法结构，不再使用 unique_grid_list）
                            for grid in unique_grids:
                                cover_dic_all_list_all_time[num_now][time][grid] += 1

            average_cover_dic_all_list_all_time = {
                num: {time: {grid: cover_dic_all_list_all_time[num][time][grid] / num_simulations for grid in grid_list}
                      for time in range(360, 1200, 60)}
                for num in range(10, car_count_initial + 1, 10)
            }

            data_time_cover = []
            for num, times in average_cover_dic_all_list_all_time.items():
                for time, grids in times.items():
                    for grid, count in grids.items():
                        data_time_cover.append({"num": num, "time": time, "grid": grid, "count": count})
            df_time_cover = pd.DataFrame(data_time_cover)

            df_time_cover.to_csv(f'./res_taxi/taxi_k/{city}/taxi_cover_count_grid{res}.csv', index=False)

            k_values = []
            edge_cover_this_station = []

            for time_key in range(360, 1200, 60):
                for edge in grid_list:
                    # 提取每个边在当前时间区间的覆盖数据
                    edge_cover_num_list = [
                        average_cover_dic_all_list_all_time.get(num, {}).get(time_key, {}).get(edge, 0)
                        for num in range(10, car_count_initial + 1, 10)
                    ]

                    if max(edge_cover_num_list) > 0:

                        edge_cover_this_station.append((edge, time_key))
                        x = np.array([i for i in range(0, car_count_initial + 1, 10)])  # 自变量
                        y = np.array([0] + edge_cover_num_list)  # 因变量

                        # 计算斜率 k
                        k = np.sum(x * y) / np.sum(x ** 2) if np.sum(x ** 2) != 0 else 0
                        # 计算预测值
                        y_pred = k * x
                        # 计算总变异
                        y_mean = np.mean(y)
                        total_variance = np.sum((y - y_mean) ** 2)

                        # 计算残差变异
                        residual_variance = np.sum((y - y_pred) ** 2)

                        # 计算 R²
                        r_squared = 1 - (residual_variance / total_variance) if total_variance != 0 else 0
                        k_values.append({'grid_id': edge, 'time_key': time_key, 'k_value': k,
                                         'r_squared': r_squared})

            # 保存 k_values 数据框
            k_values_df = pd.DataFrame(k_values)
            k_values_df.to_csv(f'./res_taxi/taxi_k/{city}/taxi_k_values_grid{res}.csv', index=False)
