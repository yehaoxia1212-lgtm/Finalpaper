from math import floor
from gurobipy import *
import pickle


def sensor_dis_model_one_day(day, N_k, station_id_list, sensor_num_end_day_dic, edge_list, edge_length_dic, start_vehicle_num_every_day_dic, k_value_dic):
    M = 1000  # 一个很大的数
    K = 1

    # 筛选这天中所有可能调度的站点对
    dis_pair_list = []
    for i in station_id_list:
        if sensor_num_end_day_dic[i] > 0:
            for j in station_id_list:
                if i != j:
                    dis_pair_list.append((i, j))

    # 创建一个模型
    Dis_model = Model("Dis_model")

    print("开始构建模型")
    # 定义决策变量
    # 每天开始时刻各站点的传感器数量
    x_start = Dis_model.addVars(station_id_list, lb=0, vtype=GRB.INTEGER, name="x_start")

    # 调度变量 (day: 调度发生的日期)
    y_dispatch = Dis_model.addVars(
        dis_pair_list,
        lb=0,
        vtype=GRB.INTEGER,
        name="y_dispatch"
    )

    # y_ew变量，表示时空路段e的覆盖是否满足需求
    y_e = Dis_model.addVars(edge_list, lb=0, vtype=GRB.BINARY, name="y_e")

    # N_ew表示路段e的期望覆盖次数
    N_e = Dis_model.addVars(edge_list, vtype=GRB.CONTINUOUS, name="N_ew")

    Dis_model.setObjective(
        quicksum(edge_length_dic[e] * y_e[e] for e in edge_list),
        GRB.MAXIMIZE
    )

    # 添加约束条件
    for node in station_id_list:
        # 当天结束时刻库存 = 开始库存 + 自然变化量
        end_inventory = sensor_num_end_day_dic[node]
        if end_inventory < 0:
            print("I AM HERE")

        in_pair = []  # 记录流入node的pair
        out_pair = []  # 记录流出node的pair
        for pair in dis_pair_list:
            if pair[1] == node:
                in_pair.append(pair)
            if pair[0] == node:
                out_pair.append(pair)

        # 调出总量
        out_flow = quicksum(y_dispatch[ss[0], ss[1]] for ss in out_pair)

        # 调入总量
        in_flow = quicksum(y_dispatch[ss[0], ss[1]] for ss in in_pair)

        # 下一天开始库存 = 当天结束库存 - 调出 + 调入
        Dis_model.addConstr(
            x_start[node] == end_inventory - out_flow + in_flow,
            name=f"balance_{node}_{day}"
        )

        # 调出不能超过可用库存
        Dis_model.addConstr(
            out_flow <= end_inventory,
            name=f"outbound_{node}_{day}"
        )

    # 传感器数量约束
    for s in station_id_list:
        Dis_model.addConstr((x_start[s] <= start_vehicle_num_every_day_dic[(day + 1, s)]), name="eq_3_22")

    Dis_model.addConstr((quicksum(x_start[s] for s in station_id_list) == N_k), name="eq_3_22")

    # 覆盖相关约束
    for e in edge_list:
        Dis_model.addConstr((N_e[e] == quicksum(k_value_dic[(s, e)] * x_start[s] for s in station_id_list)),
                            name="eq_3_23")
        Dis_model.addConstr(-M * (1 - y_e[e]) <= N_e[e] - K, name=f"cover_low_{e}")
        Dis_model.addConstr((N_e[e] - K <= M * y_e[e]), name="eq_3_24")

    # 设置求解时间限制(3600秒)
    Dis_model.setParam(GRB.Param.TimeLimit, 3600)
    Dis_model.setParam(GRB.Param.MIPGap, 0.05)

    # 7. 优化求解与诊断
    Dis_model.Params.LogToConsole = 1  # 启用求解日志
    Dis_model.optimize()

    # 结果处理
    if Dis_model.status == GRB.OPTIMAL:
        print("\n优化成功!")
        print(f"优化目标为: {Dis_model.ObjVal:.2f}")
    elif Dis_model.status == GRB.INFEASIBLE:
        print("\n模型不可行! 进行不可行性分析...")
        # 计算不可行约束
        Dis_model.computeIIS()
        Dis_model.write("infeasible_model.ilp")

    # 存储每天的单车调度过程
    sensor_dispatch_dic = {}
    for pair in dis_pair_list:
        sensor_dispatch_dic[(pair[0], pair[1], day)] = y_dispatch[pair[0], pair[1]].x

    # 存储这个字典
    # 数据保存
    # output = open("data/传感器布设优化/Dispatch/" + str(N_k) + "/" + str(day) + ".pickle", 'wb')
    # pickle.dump(sensor_dispatch_dic, output)
    # output.close()

    # 输出调度之后各站点的传感器数量
    sensor_num_next_day_dic = {}
    for s in station_id_list:
        sensor_num_next_day_dic[s] = x_start[s].x

    return sensor_num_next_day_dic, sensor_dispatch_dic



def sensor_dis_model_one_month(day, N_k, station_id_list, sensor_num_end_day_dic, edge_list, edge_length_dic, start_vehicle_num_every_day_dic, k_value_dic, current_road_coverage_dic):
    M = 1000  # 一个很大的数
    K = 1

    # 筛选这天中所有可能调度的站点对
    dis_pair_list = []
    for i in station_id_list:
        if sensor_num_end_day_dic[i] > 0:
            for j in station_id_list:
                if i != j:
                    dis_pair_list.append((i, j))

    # 创建一个模型
    Dis_model = Model("Dis_model")

    print("开始构建模型")
    # 定义决策变量
    # 每天开始时刻各站点的传感器数量
    x_start = Dis_model.addVars(station_id_list, lb=0, vtype=GRB.INTEGER, name="x_start")

    # 调度变量 (day: 调度发生的日期)
    y_dispatch = Dis_model.addVars(
        dis_pair_list,
        lb=0,
        vtype=GRB.INTEGER,
        name="y_dispatch"
    )

    # y_ew变量，表示时空路段e的覆盖是否满足需求
    y_e = Dis_model.addVars(edge_list, lb=0, vtype=GRB.BINARY, name="y_e")

    # N_ew表示路段e的期望覆盖次数
    N_e = Dis_model.addVars(edge_list, vtype=GRB.CONTINUOUS, name="N_ew")

    Dis_model.setObjective(
        quicksum(edge_length_dic[e] * y_e[e] for e in edge_list),
        GRB.MAXIMIZE
    )

    # 添加约束条件
    for node in station_id_list:
        # 当天结束时刻库存 = 开始库存 + 自然变化量
        end_inventory = sensor_num_end_day_dic[node]
        if end_inventory < 0:
            print("I AM HERE")

        in_pair = []  # 记录流入node的pair
        out_pair = []  # 记录流出node的pair
        for pair in dis_pair_list:
            if pair[1] == node:
                in_pair.append(pair)
            if pair[0] == node:
                out_pair.append(pair)

        # 调出总量
        out_flow = quicksum(y_dispatch[ss[0], ss[1]] for ss in out_pair)

        # 调入总量
        in_flow = quicksum(y_dispatch[ss[0], ss[1]] for ss in in_pair)

        # 下一天开始库存 = 当天结束库存 - 调出 + 调入
        Dis_model.addConstr(
            x_start[node] == end_inventory - out_flow + in_flow,
            name=f"balance_{node}_{day}"
        )

        # 调出不能超过可用库存
        Dis_model.addConstr(
            out_flow <= end_inventory,
            name=f"outbound_{node}_{day}"
        )

    # 传感器数量约束
    for s in station_id_list:
        Dis_model.addConstr((x_start[s] <= start_vehicle_num_every_day_dic[(day + 1, s)]), name="eq_3_22")

    Dis_model.addConstr((quicksum(x_start[s] for s in station_id_list) == N_k), name="eq_3_22")

    # 覆盖相关约束
    for e in edge_list:
        Dis_model.addConstr((N_e[e] == current_road_coverage_dic[e] + quicksum(k_value_dic[(s, e)] * x_start[s] for s in station_id_list)),
                            name="eq_3_23")
        Dis_model.addConstr(-M * (1 - y_e[e]) <= N_e[e] - K, name=f"cover_low_{e}")
        Dis_model.addConstr((N_e[e] - K <= M * y_e[e]), name="eq_3_24")

    # 设置求解时间限制(3600秒)
    Dis_model.setParam(GRB.Param.TimeLimit, 3600)
    Dis_model.setParam(GRB.Param.MIPGap, 0.05)

    # 7. 优化求解与诊断
    Dis_model.Params.LogToConsole = 1  # 启用求解日志
    Dis_model.optimize()

    # 结果处理
    if Dis_model.status == GRB.OPTIMAL:
        print("\n优化成功!")
        print(f"优化目标为: {Dis_model.ObjVal:.2f}")
    elif Dis_model.status == GRB.INFEASIBLE:
        print("\n模型不可行! 进行不可行性分析...")
        # 计算不可行约束
        Dis_model.computeIIS()
        Dis_model.write("infeasible_model.ilp")

    # 存储每天的单车调度过程
    sensor_dispatch_dic = {}
    for pair in dis_pair_list:
        sensor_dispatch_dic[(pair[0], pair[1], day)] = y_dispatch[pair[0], pair[1]].x

    # 存储这个字典
    # 数据保存
    # output = open("data/传感器布设优化/Dispatch/" + str(N_k) + "/" + str(day) + ".pickle", 'wb')
    # pickle.dump(sensor_dispatch_dic, output)
    # output.close()

    # 输出调度之后各站点的传感器数量
    sensor_num_next_day_dic = {}
    for s in station_id_list:
        sensor_num_next_day_dic[s] = x_start[s].x

    return sensor_num_next_day_dic, sensor_dispatch_dic
