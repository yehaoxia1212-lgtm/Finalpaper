# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 13:00:36 2022

@author: Meng danyang
"""



"""
成都出租车白表（6-23点）每公里1.90元.起价8元.10公里到60公里每公里2.85元；

夜表（23-6点）每公里2.2元.起价9元.10公里到60公里每公里3.30元。

起步价包含2公里运程

计价方式：500米计费5分钟按1公里计费.车速低于12公里/小时时程并计。

计价采用四舍六入五奇进整数显示。

速腾2.0和1.4T加1元。

"""


"""
input:
vel ———— 接单车辆的类
order ———— 所接订单的类
G ———— 路网

output:
vel ———— 更新完车辆（司机）收益后的车辆

"""




def dri_income (vel, order, G, in_time):
    route = order.route  #所接订单起终点间的最短路径
    length = 0  #最短路的长度初始为0
    for i in range(len(route) - 1):
        length += G[route[i]][route[i + 1]]['weight']  #计算最短路的长度
    #计算收费
    start_time = in_time[:8]
    finish_time = in_time[9:]
    if (start_time > '06:00:00') and (finish_time < '23:00:00'):  # 白天
        if length <= 2000:
            income = 8
        elif (length >= 2000) and (length <= 10000):
            income = 8 + (length - 2000) / 1000 * 1.9
            income = round(income,0)
        elif (length >= 10000) and (length <= 60000):
            income = 8 + (8 * 1.9 )+ (length - 10000) /1000 * 2.85
            income = round(income,0)

    elif ((start_time > '23:00:00') and (finish_time < '24:00:00')) or (
            (start_time > '00:00:00') and (finish_time < '06:00:00')):  # 晚上
        if length <= 2000:
            income = 9
        elif (length >= 2000) and (length <= 10000):
            income = 9 + (length - 2000) / 1000 * 2.2
            income = round(income,0)
        elif (length >= 10000) and (length <= 60000):
            income = 9 + 8 * 2.2 + (length - 10000) /1000 * 3.3
            income = round(income,0)
    #将收费计入车辆的信息中
    vel.income.append(income)

    return vel











