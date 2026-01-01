import pandas as pd
import transbigdata
import geopandas
import geopandas as gpd
from shapely.geometry import Polygon
import json
from shapely.geometry import Point
import tqdm
from shapely.wkt import loads
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from collections import Counter
from scipy.spatial.distance import pdist, squareform
import matplotlib.font_manager as fm
from matplotlib.colors import ListedColormap, BoundaryNorm
from shapely import wkt
from shapely.geometry import LineString
import requests
import warnings
import pickle
warnings.filterwarnings("ignore")

def function01(geometry):
    x,y =geometry.exterior.coords.xy
    bottom_left_lon = x[0]
    bottom_left_lat = y[0]
    top_right_lon = x[2]
    top_right_lat = y[2]
    ser1 = pd.Series([bottom_left_lon,bottom_left_lat,top_right_lon,top_right_lat])
#     arr1 = np.array([bottom_left_lon,bottom_left_lat,top_right_lon,top_right_lat])
    return  ser1

def draw_grid(data,acc):
    cq = data.copy()
    # 画1000m网格
    result = transbigdata.area_to_grid(cq, accuracy=acc, method='rect')
    # 保存到本地
    #result[0].to_file('CD{0}.json'.format(acc), driver='GeoJSON')

    lon, lat = transbigdata.grid_to_centre([result[0].iloc[:, 0], result[0].iloc[:, 1]], result[1])
    output = result[0]
    output['center_lon'] = lon
    output['center_lat'] = lat
    output[['bottom_left_lon', 'bottom_left_lat', 'top_right_lon', 'top_right_lat']] = output['geometry'].apply(
        lambda x: function01(x))
    output['grid_id'] = [i for i in range(len(output))]
    output = output[
        ['LONCOL', 'LATCOL', 'center_lon', 'center_lat', 'grid_id', 'geometry', 'bottom_left_lon', 'bottom_left_lat',
         'top_right_lon', 'top_right_lat']]

    return output
def divide_data(data):
    data['time_on'] = pd.to_datetime(data['time_on'])

    # 按日期拆分DataFrame并保存
    for date, group in data.groupby(data['time_on'].dt.date):
        # 创建一个文件名，例如，以日期命名
        filename = f"./data_taxi/data_taxi_mth_{date}.csv"
        # 保存子DataFrame到CSV文件
        group.to_csv(filename, index=False)

def node_in_zone(data_node,data_zone):
    node = data_node.copy()
    zone = data_zone.copy()
    geometry = [Point(lon, lat) for lon, lat in zip(node['x'], node['y'])]
    geo_node = gpd.GeoDataFrame(node, geometry=geometry, crs='EPSG:4326')
    geo_node = geo_node.assign(LocationID=float('nan'))

    for index, row in tqdm.tqdm(geo_node.iterrows()):
        point = row['geometry']
        intersected_grid = zone[zone['geometry'].apply(lambda poly: poly.contains(point))]['LocationID'].tolist()
        if intersected_grid:
            geo_node.at[index, 'LocationID'] = intersected_grid[0]
    geo_node['LocationID'] = geo_node['LocationID'].astype(int)
    return geo_node

def get_random_node_id(location_id, data_node):
    node = data_node.copy()
    matching_nodes = node[node['LocationID'] == location_id]['node_id']
    if not matching_nodes.empty:
        return np.random.choice(matching_nodes)
    else:
        return np.nan

# def get_node_lon_lat(node_id):
#     lon = node[node['node_id'] == node_id]['x'].values
#     lat = node[node['node_id'] == node_id]['y'].values
#     return lon[0], lat[0]
if __name__ == '__main__':
    # 获取曼哈顿公交路线表
    # data = pd.read_csv('./data_bus/shapes.txt', delimiter=',')
    # lines = data.groupby('shape_id').apply(lambda x: LineString(zip(x['shape_pt_lon'], x['shape_pt_lat'])))
    # gdf = gpd.GeoDataFrame(lines, columns=['geometry']).reset_index()
    # gdf.to_csv('./data_bus/busline_routes_Manhattan.csv')

    # 获取曼哈顿形状json文件
    # data = gpd.read_file('./data/cleaned.shp')
    # data_mht = data.iloc[[3]]
    # data_mht.to_file('./data/manhattan_shape.json', driver='GeoJSON')

    # 获取曼哈顿网格文件
    # data = gpd.read_file('./data/manhattan_shape.json')
    # grid1000 = draw_grid(data, 250)
    # grid1000.to_csv('./data/mht_grid250.csv')

    # 路网数据导入
    # node = pd.read_excel('./Manhattan_network_data/node.xls')
    # edge = pd.read_excel('./Manhattan_network_data/edge.xls')
    # adj = pd.read_excel('./Manhattan_network_data/adj.xls')
    # # 接连矩阵
    # with open('./Manhattan_network_data/adjacency_matrix.pickle', 'rb') as file:
    #     M_adj = pickle.load(file)
    # 最短路径矩阵
    # with open('./Manhattan_network_data/shortest_paths.pickle', 'rb') as file:
    #     shortest_paths_list = pickle.load(file)

    # 纽约城市和曼哈顿区域数据
    # nyc_zone = gpd.read_file('./data_taxi/taxi_zones/taxi_zones.shp')
    # nyc_zone = nyc_zone.to_crs(epsg=4326)
    # mth_zone = nyc_zone[nyc_zone['borough'] == 'Manhattan']
    # mth_zone = mth_zone.reset_index(drop=True)
    # mth_zone.to_file('./data_taxi/taxi_zones/taxi_zones_mth.json')

    # 匹配曼哈顿node和zone
    # node = node_in_zone(node, mth_zone)
    # list_zone_mth = list(set(node['LocationID'].values))
    # node.to_file('./Manhattan_network_data/node.json')
    # node.to_csv('./Manhattan_network_data/node.csv')

    # 纽约3月出租车数据
    # data_taxi = pd.read_parquet('./data_taxi/yellow_tripdata_2024-03.parquet')
    # data_taxi = data_taxi[['VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'trip_distance', 'PULocationID', 'DOLocationID']]
    # data_taxi['tpep_pickup_datetime'] = pd.to_datetime(data_taxi['tpep_pickup_datetime'])
    # data_taxi['tpep_dropoff_datetime'] = pd.to_datetime(data_taxi['tpep_dropoff_datetime'])
    # data_taxi['USING_TIME'] = (data_taxi['tpep_dropoff_datetime'] - data_taxi['tpep_pickup_datetime']).dt.total_seconds()/3600
    # data_taxi = data_taxi[data_taxi['USING_TIME'] > 0]
    # data_taxi['SPEED'] = data_taxi['trip_distance'] / data_taxi['USING_TIME']
    # average_speed = data_taxi['SPEED'].mean()



    # 匹配曼哈顿3月出租车数据
    # data_taxi_mth = data_taxi[(data_taxi['PULocationID'].isin(list_zone_mth)) & (data_taxi['DOLocationID'].isin(list_zone_mth))]
    # data_taxi_mth = data_taxi_mth.reset_index(drop=True)

    # 将原始数据中的区域id随机选择为该区域的某个节点的ID
    # data_taxi_mth['PU_node_id'] = data_taxi_mth['PULocationID'].apply(lambda x: get_random_node_id(x, node))
    # data_taxi_mth['DO_node_id'] = data_taxi_mth['DOLocationID'].apply(lambda x: get_random_node_id(x, node))
    # data_taxi_mth.to_csv('./data_taxi/mth_trip_data_2024_3.csv')

    # data_taxi_mth = pd.read_csv('./data_taxi/mth_trip_data_2024_3.csv')
    # data_taxi_mth = data_taxi_mth[['VendorID', 'tpep_pickup_datetime', 'PU_node_id', 'DO_node_id']]
    # data_taxi_mth.rename(columns={
    #     'VendorID': 'no',
    #     'tpep_pickup_datetime': 'time_on',
    #     'PU_node_id': 'node_on',
    #     'DO_node_id': 'node_off',
    # }, inplace=True)
    # data_taxi_mth[['node_on_lng', 'node_on_lat']] = data_taxi_mth['node_on'].apply(
    #     lambda x: pd.Series(get_node_lon_lat(x)))
    # data_taxi_mth[['node_off_lng', 'node_off_lat']] = data_taxi_mth['node_off'].apply(
    #     lambda x: pd.Series(get_node_lon_lat(x)))
    # data_taxi_mth.to_csv('./data_taxi/mth_trip_data_analysed.csv')

    # data_taxi_mth = pd.read_csv('./data_taxi/mth_trip_data_analysed.csv')
    # data_taxi_mth_ = data_taxi_mth[data_taxi_mth['node_on'] != data_taxi_mth['node_off']]
    # data_taxi_mth_ = data_taxi_mth_.reset_index(drop=True)
    # data_taxi_mth_.to_csv('./data_taxi/mth_trip_data_analysed.csv', index=False)
    print()




