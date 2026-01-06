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

def get_node_lon_lat(node_id):
    lon = node[node['node_id'] == node_id]['x'].values
    lat = node[node['node_id'] == node_id]['y'].values
    return lon[0], lat[0]

def find_closest_node(point, node_gdf):
    nearest_node = node_gdf.geometry.distance(Point(point)).idxmin()  # 找到距离最短的节点
    return node_gdf.loc[nearest_node, 'node_id']  # 返回节点的 ID


# 通过图 G 和公交线路的多个点来逐个匹配边
def match_edges_for_line(line_geometry, G, node_info, endpoints_of_each_edge):
    points = list(line_geometry.coords)  # 获取公交线路的所有点
    edges = []

    # 遍历所有相邻的点对
    for i in range(len(points) - 1):
        start_point = points[i]  # 起点
        end_point = points[i + 1]  # 终点

        # 找到起点和终点的最近节点
        start_node_id = find_closest_node(start_point, node_info)  # 起点最近的节点
        end_node_id = find_closest_node(end_point, node_info)  # 终点最近的节点

        # 查找起点和终点在边端点数据中的匹配
        edge = endpoints_of_each_edge[(endpoints_of_each_edge['node_id_x'] == start_node_id) &
                                      (endpoints_of_each_edge['node_id_y'] == end_node_id)]

        # 如果没有找到直接的边，尝试查找反向边（即交换起点和终点）
        if edge.empty:
            edge = endpoints_of_each_edge[(endpoints_of_each_edge['node_id_x'] == end_node_id) &
                                          (endpoints_of_each_edge['node_id_y'] == start_node_id)]

        # 如果找到边，检查是否是正确的边（即确认这对节点是否确实属于同一条路）
        if not edge.empty:
            edge_id = edge['edge_id'].values[0]
            edges.append(edge_id)

    return edges





# 定义去除连续重复节点的函数
def remove_node_duplicates(node_list):
    cleaned_list = [node_list[0]]  # 初始化，包含第一个节点
    for i in range(1, len(node_list)):
        if node_list[i] != node_list[i - 1]:  # 只保留与前一节点不同的节点
            cleaned_list.append(node_list[i])
    return cleaned_list


def node_to_edge_list(node_list, G):
    edge_list = []
    for i in range(len(node_list) - 1):
        # 获取当前节点对
        node1, node2 = node_list[i], node_list[i + 1]

        # 检查图中是否存在该边
        if G.has_edge(node1, node2):
            # 获取边的 edge_id 属性
            edge_id = G[node1][node2]['edge_id']
            edge_list.append(edge_id)
        else:
            # 如果边不存在，可以选择跳过或标记错误
            edge_list.append(None)  # 或抛出异常，取决于实际需求
    return edge_list


def bus_line_edge_simulation(data_line,node_info):

    data_line = pd.read_csv('./data_bus/busline_routes_Manhattan.csv')

    data_line['geometry'] = data_line['geometry'].apply(loads)
    node_info['geometry'] = node_info['geometry'].apply(loads)

    # 将 DataFrame 转换为 GeoDataFrame
    data_line = gpd.GeoDataFrame(data_line, geometry='geometry')
    node_info = gpd.GeoDataFrame(node_info, geometry='geometry')



    # 定义坐标转换器：从EPSG:4326到UTM
    project_to_meters = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32618",
                                                    always_xy=True).transform  # 曼哈顿常用UTM Zone 18N
    project_to_latlon = pyproj.Transformer.from_crs("EPSG:32618", "EPSG:4326", always_xy=True).transform

    # 转换geometry并创建buffer
    polygons = []

    for line in data_line['geometry']:
        # 转换到投影坐标系
        line_meters = transform(project_to_meters, line)

        # 创建5米宽度的缓冲区（半径2.5米）
        buffered_line = line_meters.buffer(10, cap_style=2)  # cap_style=2 保持端点为直线

        # 转换回地理坐标系
        buffered_line_latlon = transform(project_to_latlon, buffered_line)

        polygons.append(buffered_line_latlon)

    # 将多边形存储到新列
    data_line['polygon_geometry'] = polygons

    data_line['node_list'] = [[] for _ in range(len(data_line))]
    data_line['contained_edges'] = [[] for _ in range(len(data_line))]

    # 遍历每条线的polygon_geometry，检查包含的node
    for i, polygon in enumerate(data_line['polygon_geometry']):
        # 检查哪些点在polygon中
        contained_nodes = node_info[node_info['geometry'].apply(lambda point: polygon.contains(point))]

        # 按顺序排列节点：以polygon的顺序为准
        node_list = []
        for point in polygon.exterior.coords:  # 获取polygon的外边界坐标
            # 查找最近的节点
            nearest_node = contained_nodes.distance(Point(point)).idxmin()  # 找到距离该坐标最近的节点
            node_list.append(nearest_node)

        # 添加到node_list
        data_line.at[i, 'node_list'] = node_list

    # 应用去重逻辑
    data_line['node_list'] = data_line['node_list'].apply(remove_node_duplicates)


    def find_edges_between_nodes(node_list, edge_df):
        # 初始化一个空的边列表
        contained_edges = []

        # 遍历每对相邻的节点，检查它们是否连接
        for j in range(len(node_list) - 1):
            node_x = node_list[j]
            node_y = node_list[j + 1]

            # 检查node_x和node_y之间是否有边
            edge_ids = edge_df[
                (edge_df['node_id_x'] == node_x) & (edge_df['node_id_y'] == node_y) |
                (edge_df['node_id_x'] == node_y) & (edge_df['node_id_y'] == node_x)
                ]['edge_id'].tolist()

            # 如果有连接的路段，则加入到列表
            if edge_ids:
                contained_edges.extend(edge_ids)  # 按顺序添加路段

        return contained_edges


    # 为每条线找到包含的edges
    data_line['contained_edges'] = data_line['node_list'].apply(lambda nodes: find_edges_between_nodes(nodes, endpoints_of_each_edge))
    data_line.drop(columns=['index'], inplace=True)
    data_line['line_index'] = data_line.index
    data_line = data_line[['line_index','route_id','contained_edges','node_list','geometry','polygon_geometry','shape_id']]
    data_line.to_csv('./data_bus/busline_routes_edge.csv', index=False)
if __name__ == '__main__':


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

    #匹配曼哈顿公交线路数据
    data_shape =pd.read_csv('./shapes.txt', delimiter=',')
    lines = data_shape.groupby('shape_id').apply(lambda x: LineString(zip(x['shape_pt_lon'], x['shape_pt_lat'])))
    gdf_shape = gpd.GeoDataFrame(lines, columns=['geometry']).reset_index()
    data_trip = pd.read_csv('./trips.txt', delimiter=',')
    data_trip_unique = data_trip[['shape_id', 'route_id']].drop_duplicates()
    shape_to_route_dict = dict(zip(data_trip_unique['shape_id'], data_trip_unique['route_id']))
    gdf_shape['route_id'] = gdf_shape['shape_id'].apply(lambda x: shape_to_route_dict.get(x, None))
    list_bus = list(set(gdf_shape['route_id']))
    gdf_shape['geometry_length'] = gdf_shape.geometry.length

    # 2. 对每个 route_id 按 geometry_length 选择最长的 geometry
    gdf_shape_longest = gdf_shape.loc[gdf_shape.groupby('route_id')['geometry_length'].idxmax()]
    gdf_shape_longest.reset_index(inplace=True)
    gdf_shape_longest.to_csv('./busline_routes_Manhattan.csv',index=False)
    print()




