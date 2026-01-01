# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 13:00:36 2022

@author: Meng danyang
"""
import numpy as np

#订单完成率
"""
input:
order_list ———— 记录的订单数据

output:
order_finish_rate ———— 订单的完成率

"""

def order_finish (order_list):
    all_order_num = len(order_list)  #总订单数
    finished_order_num = 0  #完成的订单数——初始化为0
    for i in order_list:
        if i.state == 2:   #如果订单的状态为2（匹配），完成的订单数加1
            finished_order_num += 1
    order_finish_rate = finished_order_num / all_order_num  #计算订单的完成率
    return order_finish_rate



#平均等待时间
"""
input:
order_list ———— 记录的订单数据

output:
ave_wait_time ———— 平均等待时间

"""

def ave_waiting_time (order_list):
    wait_time = []
    for i in order_list:
        if i.state == 2:  #如果订单的状态为2（匹配）,添加等待时间到列表中
            wait_time.append(i.remain)
    ave_wait_time = round(np.mean(wait_time), 2)  #计算平均等待时间
    return ave_wait_time


