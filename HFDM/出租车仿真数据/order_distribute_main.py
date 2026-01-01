# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 13:00:36 2022

@author: Meng danyang
"""



import networkx as nx
import random
import numpy as np
from order_distribute_strategy import order_dis_strategy  #导入具体的订单分配策略函数
from driver_income import dri_income  #导入计算司机收入的函数

# 订单分配
"""
input:
G ———— 路网
order_list ———— 订单列表
vel_list ———— 车辆列表
vel_track_list ———— 每辆车的轨迹

output:
order_list ———— 订单分配后更新的订单列表
vel_list ———— 订单分配后更新的车辆列表
vel_track_list ———— 订单分配后更新的每辆车的轨迹

"""


def order_dis_main(G, order_list, vel_list, vel_track_list, in_time, order1):
    # 筛选出空闲的车辆
    vel_list_idle = []
    for i in vel_list:
        if i.state == 1:
            vel_list_idle.append(i)

    # 筛选出未匹配的订单
    order_list_unmatch = []
    for i in order_list:
        if i.state == 1:
            order_list_unmatch.append(i)

    # 出租车的位置
    taxi_pos = []
    for i in vel_list_idle:
        taxi_pos.append(i.loc)

    # 乘客的位置
    passenger_pos = []
    for i in order_list_unmatch:
        passenger_pos.append(i.O)

    if taxi_pos:
        if passenger_pos:
            #当既存在空车也存在未被匹配的乘客时

            # print('taxi_pos',taxi_pos)
            # print('passenger_pos',passenger_pos)

            #订单分配
            match, path_infos = order_dis_strategy(G, passenger_pos, taxi_pos)

            # 匹配结束后
            for i, match_passenger in enumerate(match):
                if match_passenger != -1:
                    # 车的更新
                    vel_list_idle[i].state = 2  # 车辆标记为载客
                    vel_list_idle[i].order = order_list_unmatch[match_passenger].no  # 标记车辆匹配的订单编号
                    vel_list_idle[i].matched_order.append(order_list_unmatch[match_passenger].no)
                    # vel_list_idle[i] = dri_income(vel_list_idle[i], order_list_unmatch[match_passenger], G, in_time)  #更新车辆（司机）的收入
                    route = path_infos[i][match_passenger]['p']  # 出租车从当前位置出发去接客的路径
                    route_min = order_list_unmatch[match_passenger].route  # 从订单起点到终点的最短路
                    vel_list_idle[i].route = route + route_min[1:]  # 车辆的路径更新
                    # 订单的更新
                    order_list_unmatch[match_passenger].state = 2  # 订单标记为已匹配
                    order_list_unmatch[match_passenger].matched_driver = vel_list_idle[i].no
                    #车辆轨迹的更新
                    vel_track_list[vel_list_idle[i].no].pas_on.append(int(order_list_unmatch[match_passenger].O)) #更新车辆轨迹中的上客点
                    vel_track_list[vel_list_idle[i].no].pas_off.append(int(order_list_unmatch[match_passenger].D))  # 更新车辆轨迹中的下客点
                    # 把已匹配订单的时间提取出来，后续过程需使用
                    number = order_list_unmatch[match_passenger].no  #得到订单号
                    pas_on_t = order1.loc[order1[order1['no'] == number].index[0]]['time_on'] #从订单总表里提取出时间
                    vel_track_list[vel_list_idle[i].no].pas_on_time.append(pas_on_t)

            # 将订单和车辆的更新状态更新到原列表中
            # vel_list_idle  ——>  vel_list
            # order_list_unmatch ——> order_list
            for i in vel_list_idle:
                vel_list[i.no] = i

            for i in order_list_unmatch:
                for j in range(len(order_list)):
                    if order_list[j].no == i.no:  # 找到相同的 no
                        order_list[j] = i  # 替换原有的对象
                        break

    return order_list, vel_list, vel_track_list