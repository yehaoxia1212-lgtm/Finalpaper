# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 13:00:36 2022

@author: Meng danyang
"""

import pandas as pd
import datetime
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
from matplotlib import rcParams
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
pd.options.display.max_columns = None
pd.options.display.max_rows = None
import copy
import networkx as nx
from road_network import network      #导入创建路网的程序
from route_plan import path_random    #导入随机生成路网的函数
from order_distribute_main import order_dis_main


class Vehicle:
    def __init__(self, no, state, order, loc, loc_index, dis, speed, route, income, order_pay, remain, R, flag):
        # 下面为vehicle对象增加实例变量
        self.no = no  # 车辆编号
        self.state = state  # 车辆状态（1：空闲；2：载客）
        self.order = order  # 车辆当前接的订单编号
        self.loc = loc  # 车辆位置（当前所在的点，如在路段上，则为上一个点）
        self.loc_index = loc_index  # 车辆位置在路径列表上的索引编号
        self.dis = dis  # 车辆距loc的距离
        self.speed = speed  # 车辆的速度
        self.route = route  # 车辆目前运动的路径
        self.income = income  # 车辆（司机）的收入
        self.order_pay = order_pay #车辆接的订单编号所对应的乘客的支出
        self.remain = remain  #司机在空驶状态保持的时间
        self.R = R  #司机最大匹配半径
        self.flag = flag  #代表司机是否具备传感能力（即是否安置传感器），1为有，0则无
        self.matched_order = []  # 司机接取的订单编号集合

    # 定义车的输出函数
    def print_fun(self):
        print("车辆编号:", self.no)
        print("车辆状态:", self.state)
        print("车辆接的订单的编号:", self.order)
        print("车辆位置:", self.loc)
        print("车辆位置的索引编号:", self.loc_index)
        print("车辆距loc的距离:", self.dis)
        print("车辆的速度:", self.speed)
        print("车辆目前运动的路径:", self.route)
        print('车辆（司机）的收入：',self.income)
        print("--------------------------------------")


# 订单（乘客）
class Order:
    def __init__(self, no, O, D, state, remain, route, route_length, t_od):
        # 下面为order对象增加实例变量
        self.no = no  # 订单编号
        self.O = O  # 订单起点
        self.D = D  # 订单终点
        self.state = state  # 订单状态（0:消失，1：未匹配，2：匹配,(删除该状态3:已完成)）
        self.remain = remain  # 订单留存时间(已保留了多久，<=3)
        self.route = route  # 订单的最短路径
        self.route_length = route_length #最短路径所需要的长度
        self.t_od = t_od  #从O到D的最短路的大致行程时间
        self.matched_driver = None

    # 定义订单的输出函数
    def print_fun(self):
        print("订单编号:", self.no)
        print("订单起点:", self.O)
        print("订单终点:", self.D)
        print("订单状态:", self.state)
        print("订单留存时间:", self.remain)
        print("订单的最短路径:", self.route)
        print("--------------------------------------")


#车辆的行驶轨迹
class vel_track:
    def __init__(self, no, track, time, track_edge, track_grid, pas_on, pas_off, pas_on_time):
        #下面为taxi_route对象增加实例变量
        self.no = no #车辆的编号
        self.track = track  #车辆的行驶轨迹(点）
        self.time = time  #车辆行驶轨迹（点）对应的时间
        self.track_edge = track_edge  #车辆的行驶轨迹（边）
        self.track_grid = track_grid  #车辆的行驶轨迹（网格）
        self.pas_on = pas_on  #车辆的上客点
        self.pas_off = pas_off  #车辆的下客点
        self.pas_on_time = pas_on_time #上车时间

    # 定义车辆行驶轨迹的输出函数
    def print_fun(self):
        print('车辆的编号：', self.no)
        print('车辆的行驶轨迹（点）：', self.track)
        print('车辆的行驶轨迹（点）对应的时间：',self.time)
        print('车辆的行驶轨迹（边）：', self.track_edge)
        print('车辆的行驶轨迹（网格）：', self.track_grid)
        print('车辆的上客点：', self.pas_on)
        print('车辆的下客点：', self.pas_off)
        print("--------------------------------------")


def order_ini(G, Order, order_list1):
    order_list0 = []
    for i in range(len(order_list1)):
        route = nx.dijkstra_path(G, order_list1.iloc[i]['node_on'], order_list1.iloc[i]['node_off'])  # 生成每个订单起终点之间的最短路,单位是KM
        # 计算这个最短路径所对应的路长和时间
        route_length = 0
        for k in range(len(route)-1):
            route_length = route_length + G[route[k]][route[k+1]]['weight']
            route_edge_id = G[route[k]][route[k + 1]]['edge_id']  #将点变成对应的边

        #计算这个订单从O到D要花多少秒
        t_od = route_length / (25*5/18)
        order_list0.append(Order(order_list1.iloc[i]['no'], order_list1.iloc[i]['node_on'], order_list1.iloc[i]['node_off'], 1, 0, route, route_length, t_od))
        # 订单的初始状态为1：未匹配，保留时间为0，以后每保留一个dt就+1，大于3了就删除，最短路径为[]
    return order_list0



def taxi_trajectory_output(date,virtual_order, vel_loc, vel_num, G, start_t, finish_t):
    '''

    :param virtual_order: 订单集, dataframe
    :param vel_loc: 车辆初始位置列表, list
    :param vel_num: 车辆数, int
    :param G: 路网图, networkx's object
    :return:
    '''
    t0 = datetime.datetime.strptime('{0} {1}:00:00'.format(date,start_t), '%Y-%m-%d %H:%M:%S')  # 开始时间
    t_end = datetime.datetime.strptime('{0} {1}:00:00'.format(date,finish_t), '%Y-%m-%d %H:%M:%S')  # 结束时间
    in_time = t0.time().strftime('%H:%M:%S') + '-' + t_end.time().strftime('%H:%M:%S')
    # 转换时间格式
    virtual_order['time_on'] = pd.to_datetime(virtual_order['time_on'])
    virtual_order['tpep_datetime'] = virtual_order['time_on']

    # 将出租车进行实例化
    vel_list = []
    for i in range(vel_num):  # 对每辆车进行初始化
        route = path_random(G, vel_loc[i].item())
        vel_list.append(Vehicle(i, 1, -1, vel_loc[i].item(), 0, 0, 25, route, [], [], 0, 3000, 1))

    # 车辆行驶轨迹的实例化
    vel_track_list = []
    for i in range(vel_num):
        # 将车辆初始位置加入到轨迹列表中
        track = []
        track.append(vel_list[i].loc)
        track_edge = []
        edge_data = G[vel_list[i].route[0]][vel_list[i].route[1]]  # 获取边的属性
        track_edge.append(edge_data['edge_id'])
        time_start = []
        time_start.append(t0)
        vel_track_list.append(vel_track(i, track, time_start, track_edge, [], [], [], []))

    # 开始仿真
    T_order = 5 # 这是派单周期（单位：秒）
    T_vel = 5  # 这是车辆的时间间隔（单位：秒）
    T_cov = 3600  # 这是覆盖周期的时间间隔(单位：秒)
    dt_order = datetime.timedelta(seconds=T_order)
    dt_vel = datetime.timedelta(seconds=T_vel)
    dt_cov = datetime.timedelta(seconds=T_cov)

    # from cyclic_update import cycle_program
    # from trajectory_processing import track_edge_grid   #导入车辆轨迹处理的函数（点——>边——>网格）
    # 仿真过程
    t_vel = t0  # 车辆仿真开始时间
    t_order = t0  # 订单仿真开始时间
    order_list = []  # 需要处理的订单信息

    benefit = []  # 记录每个派单周期司机与乘客收支情况
    final_data = []
    while t_order < t_end:
        # 现在进入正式的新派单周期
        t1_order = t_order + dt_order  # 订单分配每个循环时间为t_order--t1_order
        # 订单的生成(寻找生成时间在该循环时间区间内的订单）
        order_list1 = virtual_order[
            (virtual_order['tpep_datetime'] >= t_order) & (virtual_order['tpep_datetime'] < t1_order)]
        order_list = order_list + order_ini(G, Order, order_list1)  # 将上一决策区间的未处置订单与新订单一起在该决策区间处理
        # print(t_order)
        # 订单分配算法（分配方式：就近分配）
        order_list, vel_list, vel_track_list = order_dis_main(G, order_list, vel_list, vel_track_list, in_time,
                                                              virtual_order)

        # 订单的更新
        for i in order_list:
            if i.state == 1:  # 如果该订单未被匹配
                i.remain += 1  # 保留时间+1
                if i.remain > 3:  # 如果已经保留超过3个时间间隔则删除
                    i.state = 0
        # 司机的状态更新
        for i in vel_list:
            if i.state == 1:  # 如果司机处于空驶状态
                i.remain += 1
                if i.remain > 6:  # 如果空驶状态太久了都不匹配，可以考虑扩大匹配范围
                    i.R = 6000
            else:
                i.remain = 0
                i.R = 3000

        while t_vel < t1_order:
            t1_vel = t_vel + dt_vel  # 车辆状态更新的每个循环时间为t_vel--t1_vel
            # print(t_vel, '-', t1_vel, "时间周期结束时候的车辆状态")
            # 车的更新
            for i in vel_list:
                _ = []

                # 接下来这一段对车是否驶过路段的语句把空车巡航都考虑在内了
                # 对路径进行处理前后Index去重,因为若loc_index与loc_index+1相同，则G[i.loc][i.route[i.loc_index + 1]]不成立
                B = copy.deepcopy(i.route)
                for x in range(len(B) - 1, 0, -1):
                    if B[x] == B[x - 1]:
                        del B[x]
                i.route = B

                dx = i.speed /3.6 * T_vel  # 车辆上一个周期移动的距离,转化为M/S
                i.dis = i.dis + dx  # 更新dis
                edge_id = None
                if (i.dis) >= G[i.loc][i.route[i.loc_index + 1]]['weight']:  # 如果车辆驶过了这条道路
                    i.dis = i.dis - G[i.loc][i.route[i.loc_index + 1]]['weight']
                    i.loc_index += 1
                    i.loc = i.route[i.loc_index]
                    # 更新车辆轨迹
                    vel_track_list[i.no].track.append(i.loc)
                    vel_track_list[i.no].time.append(t1_vel)
                    # 检查车辆是否能驶过下一条道路
                    if i.loc_index + 1 < len(i.route):
                        if G.has_node(i.loc) and G.has_node(i.route[i.loc_index + 1]):
                            # 检查是否有边连接这两个节点
                            if G.has_edge(i.loc, i.route[i.loc_index + 1]):
                                edge_data = G[i.loc][i.route[i.loc_index + 1]]  # 获取边的属性
                                edge_id = edge_data['edge_id']
                        vel_track_list[i.no].track_edge.append(edge_id)
                    _.append(i.no)
                    formatted_time = t1_vel.strftime('%Y-%m-%d %H:%M:%S')
                    _.append(formatted_time)
                    _.append(i.loc)
                    _.append(edge_id)


                    final_data.append(_)

                    if i.loc_index + 1 == len(i.route):  # 如果是最后一个点
                        i.dis = 0
                        if i.state == 1:  # 如果车辆是空闲的
                            i.route = path_random(G, i.loc)  # todo 随机生成一个路径
                            i.loc_index = 0  # 车辆位置在路径列表上的索引编号更新为0

                        else:  # 如果车辆是载客的
                            i.state = 1

                            for j in range(len(order_list)):
                                if order_list[j].no == i.order:
                                    order_list[j].state = 3
                                    break
                            i.order = -1  # 车辆现在未接单，标识为-1
                            i.route = path_random(G, i.loc)  # todo 随机生成一个路径
                            i.loc_index = 0  # 车辆位置在路径列表上的索引编号更新为0
                # i.print_fun()
            # print()
            # for i in vel_list:
            #     i.print_fun()
            # print("#" * 25)

            t_vel = t1_vel  # 时间更新，进入下一个周期

        t_order = t1_order

    final_data_df = pd.DataFrame(final_data, columns=['No', 'Time','Node_now', 'edge'])
    return vel_list, order_list, vel_track_list,final_data_df


