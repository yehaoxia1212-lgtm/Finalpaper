import networkx as nx
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from collections import deque


def creat_nx_graph(edge_data, node_data, adj_data):
    G = nx.Graph()  # 创建一个图
    pos = {}  # 初始化布局信息
    # 添加节点
    for node in range(len(node_data)):
        G.add_node(node)
    # 添加边
    for i in range(len(edge_data)):
        edge_id = edge_data.loc[i, "edge_id"]
        length = edge_data.loc[i, "length"]
        result = adj_data.loc[adj_data["edge_id"] == edge_id]
        node_id_list = result["node_id"].unique()
        G.add_edge(node_id_list[0], node_id_list[1], weight=length)
    return G


def compute_and_save_shortest_paths(G, save_path=None):
    """
    计算并存储所有点对之间的最短路（优化版）

    参数:
    G -- networkx图对象
    save_path -- 结果保存路径（可选）

    返回:
    shortest_paths_dict -- 字典：{(start, end): [path1, path2, ...]}
    shortest_lengths_dict -- 字典：{(start, end): shortest_length}
    """
    # 使用字典存储结果，避免索引问题
    shortest_paths_dict = defaultdict(list)
    shortest_lengths_dict = {}

    # 先计算所有节点对的最短路径长度
    all_pairs_length = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))

    # 计算最短路径
    for start in G.nodes():
        try:
            # 使用单源最短路径算法提高效率
            paths = nx.single_source_dijkstra_path(G, start, weight='weight')
            for end, path in paths.items():
                # 存储路径
                shortest_paths_dict[(start, end)].append(path)
                # 存储长度
                shortest_lengths_dict[(start, end)] = all_pairs_length[start][end]
        except nx.NetworkXNoPath:
            # 处理不可达节点
            print(f"节点 {start} 有不可达的邻居节点")

    # 可选：保存结果
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump({
                'paths': dict(shortest_paths_dict),
                'lengths': shortest_lengths_dict
            }, f)

    return dict(shortest_paths_dict), shortest_lengths_dict


def generate_delta_paths_fast(G, O, D, delta, weight='length', max_paths=5):
    """
    优化版路径生成算法，快速找到所有长度不超过(1+delta)*最短路径的路径

    参数改进：
    - max_paths: 限制最大路径数量避免组合爆炸
    """
    # ========== 1. 预计算最短路径和长度 ==========
    try:
        shortest_length = nx.shortest_path_length(G, O, D, weight=weight)
        print("最短路长度为:", shortest_length)
        max_length = shortest_length * (1 + delta)
        print("delta路径的长度为：", max_length)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []

    # ========== 2. 双向Dijkstra加速搜索 ==========
    def bidirectional_dijkstra():
        """ 双向Dijkstra快速获取初始路径 """
        paths = []
        try:
            for path in nx.shortest_simple_paths(G, O, D, weight=weight):
                length = nx.path_weight(G, path, weight=weight)
                if length > max_length:
                    break
                paths.append((path, length))
                if len(paths) >= max_paths:
                    break
        except nx.NetworkXNoPath:
            pass
        return paths

    # ========== 3. 基于Contraction Hierarchies的加速 ==========
    def contraction_hierarchies_search():
        """ 使用CH加速路径发现（需预先构建CH） """
        try:
            from networkx.algorithms.shortest_paths.astar import astar_path
            # 使用A*的变种，启发式权重动态调整
            path = astar_path(G, O, D, heuristic=None, weight=weight)
            length = nx.path_weight(G, path, weight=weight)
            return [(path, length)] if length <= max_length else []
        except:
            return bidirectional_dijkstra()

    # ========== 4. 并行化路径生成 ==========
    def parallel_path_generation():
        """ 使用多线程生成路径（示例） """
        from concurrent.futures import ThreadPoolExecutor
        paths = []
        with ThreadPoolExecutor() as executor:
            futures = []
            for w in [0.5, 0.3, 0.1]:  # 不同启发式权重
                futures.append(executor.submit(
                    nx.astar_path, G, O, D,
                    heuristic=lambda u, v: w * heuristic(u, v),
                    weight=weight
                ))
            for future in futures:
                try:
                    path = future.result()
                    length = nx.path_weight(G, path, weight=weight)
                    if length <= max_length:
                        paths.append((path, length))
                except:
                    continue
        return paths

    # ========== 5. 启发式函数优化 ==========
    def heuristic(u, v):
        """ 缓存启发式值加速计算 """
        if not hasattr(G, '_heuristic_cache'):
            G._heuristic_cache = {}
        key = (u, v)
        if key not in G._heuristic_cache:
            try:
                pos_u = G.nodes[u]['pos']
                pos_v = G.nodes[v]['pos']
                G._heuristic_cache[key] = ((pos_u[0] - pos_v[0]) ** 2 + (pos_u[1] - pos_v[1]) ** 2) ** 0.5
            except KeyError:
                try:
                    G._heuristic_cache[key] = nx.shortest_path_length(G, u, v, weight=weight)
                except:
                    G._heuristic_cache[key] = 0
        return G._heuristic_cache[key]

    # ========== 主逻辑 ==========
    valid_paths = []

    # 策略1：优先使用双向Dijkstra获取短路径
    valid_paths.extend(bidirectional_dijkstra())

    # 策略2：补充长路径（限制数量）
    if len(valid_paths) < max_paths:
        valid_paths.extend(contraction_hierarchies_search())

    # 去重并排序
    unique_paths = []
    seen = set()
    for path, length in sorted(valid_paths, key=lambda x: -x[1]):
        path_tuple = tuple(path)
        if path_tuple not in seen:
            seen.add(path_tuple)
            unique_paths.append((path, length))
            if len(unique_paths) >= max_paths:
                break

    return unique_paths


def generate_delta_paths(G, O, D, delta, weight='length'):
    """
    生成起点O到终点D之间，长度不超过最短路长度(1+delta)倍的所有路径，
    并优先返回长度接近最大允许长度的路径

    参数:
    G -- networkx图对象
    O -- 起点节点
    D -- 终点节点
    delta -- 路径长度容忍系数
    weight -- 边权重属性（默认为'length'）

    返回:
    valid_paths -- 所有满足长度要求的路径列表，按长度降序排序
    """
    # 1. 计算最短路径长度
    try:
        # 计算最短路
        shortest_path = nx.shortest_path(G, source=O, target=D, weight=weight, method='dijkstra')
        # print("最短路为：", shortest_path)
        shortest_length = nx.shortest_path_length(G, source=O, target=D, weight=weight)
        # print("最短路长度为：", shortest_length)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []  # 没有可行路径或节点不存在

    max_length = shortest_length * (1 + delta)
    # print("delta最短路长度为：", max_length)

    # 2. 使用带偏差的A*算法
    valid_paths = []

    # 定义启发式函数（使用欧氏距离或预计算的最短距离）
    def heuristic(u, v):
        try:
            # 如果节点有坐标信息，使用欧氏距离
            pos_u = G.nodes[u]['pos']
            pos_v = G.nodes[v]['pos']
            return ((pos_u[0] - pos_v[0]) ** 2 + (pos_u[1] - pos_v[1]) ** 2) ** 0.5
        except KeyError:
            # 否则使用最短路径长度作为启发值
            try:
                return nx.shortest_path_length(G, u, v, weight=weight)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return 0  # 如果不可达，返回0

    # 使用带偏差的A*算法寻找路径
    # 增加启发式权重会使路径更长
    for w in [0.5, 0.3, 0.1]:  # 较小的权重会产生更长的路径
        try:
            # 修复：lambda函数需要接受两个参数（当前节点和目标节点）
            path = nx.astar_path(G, O, D, heuristic=lambda u, v: w * heuristic(u, v), weight=weight)
            path_length = nx.path_weight(G, path, weight=weight)
            if path_length <= max_length:
                valid_paths.append((path, path_length))
        except nx.NetworkXNoPath:
            continue

    # 3. 使用Yen's算法补充更多路径
    try:
        # 获取所有不超过最大长度的路径
        for path in nx.shortest_simple_paths(G, O, D, weight=weight):
            try:
                path_length = nx.path_weight(G, path, weight=weight)
                if path_length > max_length:
                    break
                # 检查是否已存在相同路径
                path_tuple = tuple(path)
                if not any(tuple(p) == path_tuple for p, _ in valid_paths):
                    valid_paths.append((path, path_length))
            except (KeyError, nx.NetworkXError):
                continue
    except nx.NetworkXNoPath:
        pass

    # 4. 按长度降序排序（最长的在前面）
    valid_paths.sort(key=lambda x: x[1], reverse=True)

    # 5. 移除重复路径
    unique_paths = []
    seen = set()
    for path, length in valid_paths:
        path_tuple = tuple(path)
        if path_tuple not in seen:
            seen.add(path_tuple)
            unique_paths.append((path, length))
    # print("delta最短路列表为：", unique_paths)
    # 只输出最长的那个路径,索引为0的位置对应的是路径列表，索引为1的位置对应的是路径长度
    return unique_paths


# 给定一个点的序列node_route，计算这个序列的长度
def calculate_node_route_length(node_route, edge_data, adjacency_matrix):
    length = 0
    for i in range(len(node_route) - 1):
        edge_id = int(adjacency_matrix[node_route[i]][node_route[i + 1]])
        edge_length = edge_data.loc[edge_id, "length"]
        length += edge_length
    return length


def convert_node_list_to_edge_list(node_route, adjacency_matrix):
    edge_route_list = []
    for i in range(len(node_route) - 1):
        edge_id = int(adjacency_matrix[node_route[i]][node_route[i + 1]])
        edge_route_list.append(edge_id)
    return edge_route_list


def calculate_shortest_distance(node_1, node_2, shortest_paths_list, edge_data, adjacency_matrix):
    shortest_route = shortest_paths_list[node_1][node_2][0]
    distance = calculate_node_route_length(shortest_route, edge_data, adjacency_matrix)
    return distance

