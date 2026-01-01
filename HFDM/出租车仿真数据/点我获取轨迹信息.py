import pandas as pd
from road_network import network
from main_simulation import taxi_trajectory_output
import random
import datetime
import pickle
from datetime import datetime, timedelta
if __name__ == '__main__':
    node_info = pd.read_csv('./Manhattan_network_data/node.csv')  # 读取路网数据——点
    edge_info = pd.read_excel('./Manhattan_network_data/edge.xls')  # 读取路网数据——边的标号和长度
    endpoints_of_each_edge = pd.read_csv('./Manhattan_network_data/endpoints_of_each_edge.csv')  # 读取路网数据——点和边之间的关系
    list_mth_zones = list(set(node_info['node_id'].values))
    # 构建路网
    # 路网的实例化
    node_id = node_info['node_id'].tolist()  # 点的编号
    node_id_x = endpoints_of_each_edge['node_id_x'].tolist()  # 每条边的一个节点
    node_id_y = endpoints_of_each_edge['node_id_y'].tolist()  # 每条边的另一个节点
    length = edge_info['length'].tolist()  # 每条边的长度
    edge_id = edge_info['edge_id'].tolist()  # 每条边的编号
    # 创建路网
    G = network(node_id, node_id_x, node_id_y, length, edge_id)

    start_date = datetime(2024, 3, 1)
    end_date = datetime(2024, 3, 1)
    # 生成日期列表
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date.strftime("%Y-%m-%d"))  # 转换为字符串格式
        current_date += timedelta(days=1)
    for date in date_list:
        virtual_order = pd.read_csv('./data_taxi_mth_{0}.csv'.format(date))
        print(date)
        start_t = 6  # 该范围为仿真时间范围
        finish_t = 20
        vel_num = 500  # 车的总数
        vel_loc = random.choices(list_mth_zones, k=500)

        # 初始化时，给定每辆车的位置（路网节点）
        # todo 2024.10.05: 输出的是三个类对象。轨迹信息放在了vel_track_list的track属性中，由一系列节点id构成（包括订单走行路径+随机巡游）
        vel_list, order_list, vel_track_list,final_data = taxi_trajectory_output(date, virtual_order, vel_loc, vel_num, G, start_t, finish_t)
        print('输出车辆的轨迹:')
        for i in vel_track_list:
            i.print_fun()
            no_taxi = i.no
            with open('./仿真车辆数据/{0}_taxi_no_{1}.pkl'.format(date,no_taxi), 'wb') as f:
                pickle.dump(i, f)
        final_data.to_csv('./仿真车辆数据/taxi_result_{0}.csv'.format(date))

        print("#" * 25)
