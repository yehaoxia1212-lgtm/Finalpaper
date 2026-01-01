# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 13:00:36 2022

@author: Meng danyang
"""

"""
import:
vel_track_list ———— 车辆轨迹（点）
edge_grid_pairs ———— 路网中边和网格的关系
G ———— 路网

output:
vel_track_list ———— 车辆轨迹（点、边、网格（去重））

"""

from grid_coverage import grid_cov  #导入网格覆盖次数的计算函数


def track_edge_grid (t0, t_end, dt_cov, vel_track_list, edge_grid_pairs, grid_id, G):
    t_cov_end = []  #记录每个覆盖更新的时间节点
    t0_start = t0
    while t0_start < t_end:  #如果在决策区间内
        dt = t_end - t0_start  #剩余决策时间间隔
        if dt_cov < dt:  #如果剩余决策时间间隔还大于一个覆盖更新时间间隔
            t_cov_end.append(t0_start + dt_cov)
            t0_start = t0_start + dt_cov
        else:  #如果剩余决策时间间隔小于等于一个覆盖更新时间间隔
            t_cov_end.append(t_end)
            t0_start = t_end
    # print('t_cov_end',t_cov_end)

    #对于每个覆盖更新的时间段，分别计算覆盖率
    # 改动：如果车辆轨迹对应的时间节点有跨过覆盖周期的，必须要提取出来
    # 将之放到下一个覆盖周期里
    index_start = 0
    for t_finish in t_cov_end:
        # print('t_finish', t_finish)
        for i in vel_track_list:  #对于每辆车
            track_grid_list = []
            selected = [x for x in i.time if x <= t_finish] #找到在覆盖更新时间段内的时间记录
            # print('selected', selected)
            track = i.track[index_start:len(selected)] #找到覆盖更新时间段内的轨迹记录
            # print('track', track)
            for j in range(len(track) - 1):
                track_edge_id = G[track[j]][track[j + 1]]['edge_id']  #将点变成对应的边
                track_grid_id = edge_grid_pairs[edge_grid_pairs['edge_id'] == track_edge_id]['grid_id']  #将边变成对应的网格的唯一标识符
                i.track_edge.append(track_edge_id)   #将边添加到行驶轨迹（边）中
                for k in track_grid_id:
                    track_grid_no = grid_id[grid_id['grid_id'] == k].iloc[0]['no']  # 根据网格的唯一标识符转变为连续编号
                    track_grid_list.append(track_grid_no)   #将网格以连续编号的形式记录在列表中
            #网格去重处理
            new_list = sorted(set(track_grid_list), key=track_grid_list.index)  #去重
            i.track_grid.append(new_list)  # 将去重后的网格以连续编号的形式记录在行驶轨迹（网格）中
        index_start = len(selected)
    return vel_track_list







