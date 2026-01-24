# -*- coding: utf-8 -*-
import os
import json
import math
from functools import lru_cache

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point

# =========================
# 0) 配置：改这里
# =========================
EDGE_GEOJSON = r"D:\Finalpaper\LFDM\longquanyi\roadmap_taxi_longquanyi\edge.geojson"
ADJ_CSV      = r"D:\Finalpaper\LFDM\longquanyi\roadmap_taxi_longquanyi\adj_matrix.csv"
TAXI_CSV     = r"D:\Finalpaper\LFDM\longquanyi\data_taxi\data_taxi_longquanyi.csv"

CAB_COL  = "cab_id"
TIME_COL = "date_time"
LON_COL  = "lon"
LAT_COL  = "lat"

U_COL   = "node_id_x"
V_COL   = "node_id_y"
EID_COL = "edge_id"

# trip segmentation（时间/速度断开）
TIME_GAP_MIN  = 10
MAX_SPEED_KMH = 130

# 离网判定：点到最近边距离 > DROP_FAR_M 就认为离网（直接切 trip）
DROP_FAR_M    = 250
SEARCH_RMAX_M = 1200

# 候选边
K_CAND     = 12
MAX_SNAP_M = 80
SIGMA_Z    = 15.0
LAMBDA_TR  = 0.15
VMAX_MPS   = 40.0

# 补全安全阈值（防止跨城补出一大坨）
MAX_FILL_EDGES = 800
MAX_FILL_DIST  = 8000.0  # 图权重为 length_m，通常单位是米

# 输出目录
OUT_DIR = r"D:\Finalpaper\LFDM\longquanyi\data_taxi\final_clean_trips"
os.makedirs(OUT_DIR, exist_ok=True)

# 总输出（稳定 JSONL：一车一行）
OUT_ALL_JSONL = os.path.join(OUT_DIR, "final_all_cabs.jsonl")

# 全体点级合并（可选）
SAVE_ALL_POINTS_CSV = True
OUT_ALL_POINTS_CSV = os.path.join(OUT_DIR, "edge_hmm_all_cabs_all_points.csv")

# 只跑某辆车调试用
ONLY_CAB = None  # e.g. 2


# =========================
# 1) 投影辅助
# =========================
def ensure_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        return gdf.set_crs("EPSG:4326", allow_override=True)
    if str(gdf.crs).upper() != "EPSG:4326":
        return gdf.to_crs("EPSG:4326")
    return gdf

def choose_utm_epsg_from_wgs(gdf_wgs: gpd.GeoDataFrame) -> int:
    lon = gdf_wgs.geometry.unary_union.centroid.x
    lat = gdf_wgs.geometry.unary_union.centroid.y
    zone = int((lon + 180) // 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone

def project_fixed(gdf_wgs: gpd.GeoDataFrame, epsg: int) -> gpd.GeoDataFrame:
    gdf_wgs = ensure_wgs84(gdf_wgs)
    return gdf_wgs.to_crs(epsg=epsg)


# =========================
# 2) 读路网：edges_m + sindex + Graph + eid2ends + eid2uv
# =========================
def load_network(edge_geojson, adj_csv):
    edges = gpd.read_file(edge_geojson)[[EID_COL, "geometry"]]
    edges = ensure_wgs84(edges)

    fixed_epsg = choose_utm_epsg_from_wgs(edges)
    edges_m = project_fixed(edges, fixed_epsg)
    edges_m["length_m"] = edges_m.geometry.length
    sidx = edges_m.sindex

    adj = pd.read_csv(adj_csv)
    if {U_COL, V_COL, EID_COL}.issubset(adj.columns):
        adj = adj.rename(columns={U_COL: "u", V_COL: "v", EID_COL: "edge_id"})
    else:
        if not {"u", "v", "edge_id"}.issubset(adj.columns):
            raise ValueError("adj_matrix.csv 缺少 node_id_x/node_id_y/edge_id 或 u/v/edge_id")

    adj["u"] = pd.to_numeric(adj["u"], errors="raise").astype(int)
    adj["v"] = pd.to_numeric(adj["v"], errors="raise").astype(int)
    adj["edge_id"] = pd.to_numeric(adj["edge_id"], errors="raise").astype(int)

    eid2ends = {}
    eid2uv = {}
    for _, r in adj.iterrows():
        eid = int(r["edge_id"])
        u, v = int(r["u"]), int(r["v"])
        eid2ends.setdefault(eid, set()).add((u, v))
        eid2uv.setdefault(eid, []).append((u, v))

    # 权重（用几何长度）
    len_map = dict(zip(edges_m[EID_COL].astype(int).values, edges_m["length_m"].values))
    adj["w"] = adj["edge_id"].astype(int).map(len_map).fillna(1.0).astype(float)

    G = nx.Graph()
    for _, r in adj.iterrows():
        u, v = int(r["u"]), int(r["v"])
        w = float(r["w"])
        eid = int(r["edge_id"])
        if G.has_edge(u, v):
            if w < G[u][v].get("weight", float("inf")):
                G[u][v].update(weight=w, edge_id=eid)
        else:
            G.add_edge(u, v, weight=w, edge_id=eid)

    return edges_m, sidx, G, eid2ends, eid2uv, fixed_epsg


# =========================
# 3) 点到最近边距离
# =========================
def nearest_edge_dist_m(p, edges_m, sidx, r_list):
    for r in r_list:
        buf = p.buffer(float(r))
        try:
            ilocs = list(sidx.query(buf, predicate="intersects"))
        except TypeError:
            ilocs = list(sidx.query(buf))
        if not ilocs:
            continue
        return min(float(edges_m.geometry.iloc[int(i)].distance(p)) for i in ilocs)
    return float("inf")

def compute_dist2net(gdf_wgs, edges_m, sidx, fixed_epsg, search_rmax_m=1200):
    gdf_m = project_fixed(gdf_wgs, fixed_epsg)

    r_list = []
    r = max(80, int(MAX_SNAP_M))
    while r < search_rmax_m:
        r_list.append(r)
        r = int(r * 2)
    r_list.append(int(search_rmax_m))

    dists = [nearest_edge_dist_m(p, edges_m, sidx, r_list) for p in gdf_m.geometry]
    return np.array(dists, dtype=float)


# =========================
# 4) 离网切 trip + 时间/速度切 trip
# =========================
def split_trips_by_network(gdf_wgs, fixed_epsg, dist2net_m, drop_far_m=250):
    gm = project_fixed(gdf_wgs, fixed_epsg)
    xy = np.c_[gm.geometry.x.values, gm.geometry.y.values]
    times = pd.to_datetime(gdf_wgs[TIME_COL]).tolist()

    onnet = np.isfinite(dist2net_m) & (dist2net_m <= float(drop_far_m))

    trips = []
    cur = []
    for i in range(len(gdf_wgs)):
        if not onnet[i]:
            if cur:
                trips.append(np.array(cur, dtype=int))
                cur = []
            continue

        if not cur:
            cur = [i]
            continue

        prev_i = cur[-1]
        dt_min = (times[i] - times[prev_i]).total_seconds() / 60.0
        if dt_min <= 0:
            dt_min = 1e-6
        dist_m = float(np.linalg.norm(xy[i] - xy[prev_i]))
        speed_kmh = (dist_m / 1000.0) / (dt_min / 60.0)

        if (dt_min > TIME_GAP_MIN) or (speed_kmh > MAX_SPEED_KMH):
            trips.append(np.array(cur, dtype=int))
            cur = [i]
        else:
            cur.append(i)

    if cur:
        trips.append(np.array(cur, dtype=int))

    return trips


# =========================
# 5) 候选边
# =========================
def build_candidates(trip_wgs, edges_m, sidx, fixed_epsg, K=12, max_snap_m=80):
    trip_m = project_fixed(trip_wgs, fixed_epsg)
    cand_edges, cand_dists = [], []
    xy = np.c_[trip_m.geometry.x.values, trip_m.geometry.y.values]

    for p in trip_m.geometry:
        buf = p.buffer(float(max_snap_m))
        try:
            ilocs = list(sidx.query(buf, predicate="intersects"))
        except TypeError:
            ilocs = list(sidx.query(buf))

        if not ilocs:
            cand_edges.append([])
            cand_dists.append([])
            continue

        tmp = {}
        for iloc in ilocs:
            line = edges_m.geometry.iloc[int(iloc)]
            d = float(line.distance(p))
            if d <= float(max_snap_m):
                eid = int(edges_m.iloc[int(iloc)][EID_COL])
                if (eid not in tmp) or (d < tmp[eid]):
                    tmp[eid] = d

        if not tmp:
            cand_edges.append([])
            cand_dists.append([])
            continue

        items = sorted(tmp.items(), key=lambda x: x[1])[:int(K)]
        cand_edges.append([int(e) for e, _ in items])
        cand_dists.append([float(d) for _, d in items])

    return cand_edges, cand_dists, xy


# =========================
# 6) HMM（Viterbi）
# =========================
def endpoints_of_edge(eid2ends, eid):
    ends = eid2ends.get(int(eid), None)
    if not ends:
        return []
    nodes = set()
    for u, v in ends:
        nodes.add(int(u)); nodes.add(int(v))
    return list(nodes)

_G_REG = {}

@lru_cache(maxsize=250000)
def sp_len_hmm(G_id, a, b):
    G = _G_REG[G_id]
    if a == b:
        return 0.0
    try:
        return float(nx.shortest_path_length(G, a, b, weight="weight"))
    except nx.NetworkXNoPath:
        return float("inf")

def edge_to_edge_netdist(G_id, eid2ends, e1, e2):
    e1 = int(e1); e2 = int(e2)
    if e1 == e2:
        return 0.0
    A = endpoints_of_edge(eid2ends, e1)
    B = endpoints_of_edge(eid2ends, e2)
    if not A or not B:
        return float("inf")
    best = float("inf")
    for a in A:
        for b in B:
            d = sp_len_hmm(G_id, a, b)
            if d < best:
                best = d
    return best

def viterbi_one_segment(G_id, eid2ends, C, D, XY, TT, sigma_z, lambda_tr, vmax_mps):
    T2 = len(C)
    if T2 == 0:
        return []

    dp, bp = [], []
    dp0 = [(d / sigma_z) ** 2 for d in D[0]]
    bp0 = [-1 for _ in D[0]]
    dp.append(dp0); bp.append(bp0)

    for t in range(1, T2):
        dt = (TT[t] - TT[t-1]).total_seconds()
        if dt <= 0:
            dt = 1.0
        euclid = float(np.linalg.norm(XY[t] - XY[t-1]))

        cur_dp = [float("inf")] * len(C[t])
        cur_bp = [-1] * len(C[t])

        for i2, e_to in enumerate(C[t]):
            emit = (D[t][i2] / sigma_z) ** 2
            best_cost, best_j = float("inf"), -1
            for j2, e_from in enumerate(C[t-1]):
                prev = dp[t-1][j2]
                net = edge_to_edge_netdist(G_id, eid2ends, e_from, e_to)
                if math.isinf(net):
                    continue
                vmax_allow = vmax_mps * dt
                speed_pen = 0.0
                if net > vmax_allow:
                    speed_pen = (net - vmax_allow) / 50.0
                tr = lambda_tr * abs(net - euclid) / 50.0 + speed_pen
                cost = prev + tr + emit
                if cost < best_cost:
                    best_cost, best_j = cost, j2
            cur_dp[i2] = best_cost
            cur_bp[i2] = best_j

        dp.append(cur_dp)
        bp.append(cur_bp)

    last_i = int(np.argmin(dp[-1]))
    best_edges = [None] * T2
    best_edges[-1] = C[-1][last_i]
    idx = last_i
    for t in range(T2 - 1, 0, -1):
        idx = bp[t][idx]
        if idx < 0:
            break
        best_edges[t-1] = C[t-1][idx]

    return [int(e) if e is not None else None for e in best_edges]

def hmm_match_split_by_cand(G_id, eid2ends, cand_edges, cand_dists, xy, times):
    valid = [len(c) > 0 for c in cand_edges]
    out, ranges = [], []
    i, T = 0, len(cand_edges)
    while i < T:
        if not valid[i]:
            i += 1
            continue
        j = i
        while j < T and valid[j]:
            j += 1
        C = cand_edges[i:j]
        D = cand_dists[i:j]
        XY = xy[i:j]
        TT = times[i:j]
        matched = viterbi_one_segment(G_id, eid2ends, C, D, XY, TT, SIGMA_Z, LAMBDA_TR, VMAX_MPS)
        out.append(matched)
        ranges.append((i, j))
        i = j
    return out, ranges


# =========================
# 7) 补全：最短路插边（用压缩后的 edge_seq）
# =========================
_G_FILL = None

@lru_cache(maxsize=400000)
def sp_len_fill(a, b):
    if a == b:
        return 0.0
    try:
        return float(nx.shortest_path_length(_G_FILL, a, b, weight="weight"))
    except nx.NetworkXNoPath:
        return float("inf")

def endpoints_nodes(eid2uv, eid):
    cands = eid2uv.get(int(eid), [])
    nodes = set()
    for u, v in cands:
        nodes.add(int(u)); nodes.add(int(v))
    return list(nodes)

def best_endpoint_pair(eid2uv, e1, e2):
    A = endpoints_nodes(eid2uv, e1)
    B = endpoints_nodes(eid2uv, e2)
    if not A or not B:
        return None, None, float("inf")
    best_d = float("inf")
    best_a, best_b = None, None
    for a in A:
        for b in B:
            d = sp_len_fill(int(a), int(b))
            if d < best_d:
                best_d, best_a, best_b = d, int(a), int(b)
    return best_a, best_b, float(best_d)

def node_path_to_edge_ids(node_path):
    eids = []
    for u, v in zip(node_path[:-1], node_path[1:]):
        data = _G_FILL.get_edge_data(u, v)
        if not data:
            continue
        eid = data.get("edge_id", None)
        if eid is None:
            continue
        eid = int(eid)
        if not eids or eids[-1] != eid:
            eids.append(eid)
    return eids

def fill_edges_shortest_path(eid2uv, edge_list):
    edges = [int(e) for e in edge_list if e is not None]
    if not edges:
        return []

    filled = [edges[0]]
    for e1, e2 in zip(edges[:-1], edges[1:]):
        if e2 == e1:
            continue

        a, b, d = best_endpoint_pair(eid2uv, e1, e2)
        if a is None or b is None or math.isinf(d) or (d > float(MAX_FILL_DIST)):
            if filled[-1] != e2:
                filled.append(e2)
            continue

        try:
            node_path = nx.shortest_path(_G_FILL, a, b, weight="weight")
        except nx.NetworkXNoPath:
            if filled[-1] != e2:
                filled.append(e2)
            continue

        mid = node_path_to_edge_ids(node_path)
        if len(mid) > int(MAX_FILL_EDGES):
            if filled[-1] != e2:
                filled.append(e2)
            continue

        for eid in mid:
            if filled[-1] != eid:
                filled.append(eid)
        if filled[-1] != e2:
            filled.append(e2)

    out = []
    for e in filled:
        if not out or out[-1] != e:
            out.append(int(e))
    return out

def compress_edges(point_edges):
    seq = []
    for e in point_edges:
        if e is None:
            continue
        e = int(e)
        if not seq or seq[-1] != e:
            seq.append(e)
    return seq


# =========================
# 8) 去刺头：节点连续性
# =========================
def choose_uv_for_edge(eid2uv, eid, cur_node=None, next_eid=None):
    cands = eid2uv.get(int(eid), [])
    if not cands:
        return None
    if cur_node is not None:
        for u, v in cands:
            if cur_node == u or cur_node == v:
                return (int(u), int(v))
        return (int(cands[0][0]), int(cands[0][1]))
    if next_eid is not None:
        nxt = eid2uv.get(int(next_eid), [])
        nxt_nodes = set()
        for a, b in nxt:
            nxt_nodes.add(int(a)); nxt_nodes.add(int(b))
        for u, v in cands:
            if int(u) in nxt_nodes or int(v) in nxt_nodes:
                return (int(u), int(v))
    return (int(cands[0][0]), int(cands[0][1]))

def remove_spurs_by_node_continuity(edge_seq, eid2uv):
    if not edge_seq:
        return []
    node_stack, edge_stack = [], []
    i = 0
    while i < len(edge_seq):
        eid = int(edge_seq[i])
        nxt = int(edge_seq[i + 1]) if i + 1 < len(edge_seq) else None

        if not node_stack:
            uv = choose_uv_for_edge(eid2uv, eid, cur_node=None, next_eid=nxt)
            if uv is None:
                i += 1
                continue
            u, v = uv
            node_stack = [u, v]
            edge_stack.append(eid)
            i += 1
            continue

        cur = node_stack[-1]
        uv = choose_uv_for_edge(eid2uv, eid, cur_node=cur, next_eid=nxt)
        if uv is None:
            i += 1
            continue
        u, v = uv

        if cur == u or cur == v:
            nxt_node = v if cur == u else u
            node_stack.append(nxt_node)
            edge_stack.append(eid)
            i += 1
            continue

        # 尝试删上一条边（毛刺）
        if len(node_stack) >= 2:
            prev = node_stack[-2]
            if prev == u or prev == v:
                if edge_stack:
                    edge_stack.pop()
                node_stack.pop()
                cur2 = node_stack[-1]
                uv2 = choose_uv_for_edge(eid2uv, eid, cur_node=cur2, next_eid=nxt)
                if uv2 is not None:
                    u2, v2 = uv2
                    if cur2 == u2 or cur2 == v2:
                        nxt_node2 = v2 if cur2 == u2 else u2
                        node_stack.append(nxt_node2)
                        edge_stack.append(eid)
                i += 1
                continue

        # 断裂：重开（不清空已累计边）
        node_stack = []
        i += 1

    clean = []
    for e in edge_stack:
        if not clean or clean[-1] != e:
            clean.append(int(e))
    return clean


# =========================
# 9) 时间字段
# =========================
def safe_date_str(dt):
    s = str(dt)
    if " " in s:
        return s.split(" ")[0]
    if "T" in s:
        return s.split("T")[0]
    return s[:10]


# =========================
# 10) 跑一辆车：输出 trips（最终补全+去刺头） + points_df（点级原始匹配）
# =========================
def process_one_cab(df, cab_id, edges_m, sidx, eid2ends, eid2uv, fixed_epsg, G_id):
    sub = df[df[CAB_COL] == cab_id].copy()
    if sub.empty:
        return None, None

    gdf = gpd.GeoDataFrame(
        sub,
        geometry=[Point(xy) for xy in zip(sub[LON_COL].astype(float), sub[LAT_COL].astype(float))],
        crs="EPSG:4326"
    ).sort_values(TIME_COL).copy()

    # 点级列：dist2net / trip_no / matched_edge_id
    gdf["dist2net_m"] = np.nan
    gdf["trip_no"] = pd.array([None] * len(gdf), dtype="Int64")
    gdf["matched_edge_id"] = pd.array([None] * len(gdf), dtype="Int64")

    # dist2net（全点）
    dist2net = compute_dist2net(gdf, edges_m, sidx, fixed_epsg, SEARCH_RMAX_M)
    gdf["dist2net_m"] = dist2net

    # 离网切 trip + 时间/速度切 trip
    trip_groups = split_trips_by_network(gdf, fixed_epsg, dist2net, DROP_FAR_M)

    out = {"cab_id": str(cab_id), "trips": []}

    trip_no = 0

    for idxs in trip_groups:
        if len(idxs) < 2:
            continue

        trip_pts = gdf.iloc[idxs].copy().sort_values(TIME_COL)
        times = pd.to_datetime(trip_pts[TIME_COL]).tolist()

        # 候选
        cand_edges, cand_dists, xy = build_candidates(trip_pts, edges_m, sidx, fixed_epsg, K_CAND, MAX_SNAP_M)

        # cand=[] 再切
        matched_list, ranges = hmm_match_split_by_cand(G_id, eid2ends, cand_edges, cand_dists, xy, times)
        if not matched_list:
            continue

        for seg_i, matched in enumerate(matched_list):
            if not matched:
                continue

            s, e = ranges[seg_i]               # s,e 在 trip_pts 内
            idx_in_gdf = trip_pts.index[s:e]   # 对回 gdf 的真实 index

            # 新trip编号
            trip_no += 1
            gdf.loc[idx_in_gdf, "trip_no"] = int(trip_no)

            # 点级匹配（原始：不补全不去刺头）
            gdf.loc[idx_in_gdf, "matched_edge_id"] = pd.array(matched, dtype="Int64")

            # trip 的时间
            start_time = times[s]
            end_time = times[e - 1]

            # —— 最终轨迹：压缩 -> 补全 -> 去刺头 —— #
            edge_seq = compress_edges(matched)
            filled = fill_edges_shortest_path(eid2uv, edge_seq)
            cleaned = remove_spurs_by_node_continuity(filled, eid2uv)

            if not cleaned:
                continue

            out["trips"].append({
                "trip_no": int(trip_no),
                "date": safe_date_str(start_time),
                "start_time": str(start_time),
                "end_time": str(end_time),
                "edge_seq": [int(x) for x in cleaned]
            })

    points_df = gdf.drop(columns=["geometry"]).sort_values(TIME_COL).copy()
    return out, points_df


# =========================
# 11) 主程序
# =========================
def main():
    global _G_FILL

    edges_m, sidx, G, eid2ends, eid2uv, fixed_epsg = load_network(EDGE_GEOJSON, ADJ_CSV)
    _G_FILL = G
    sp_len_fill.cache_clear()

    G_id = "G0"
    _G_REG[G_id] = G
    sp_len_hmm.cache_clear()

    df = pd.read_csv(TAXI_CSV)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])

    df[CAB_COL] = df[CAB_COL].astype(str).str.strip()

    cab_ids = sorted(
        df.loc[df[CAB_COL].notna() & df[CAB_COL].ne("") & df[CAB_COL].ne("nan"), CAB_COL]
        .unique()
        .tolist()
    )

    if ONLY_CAB is not None:
        ONLY_CAB_STR = str(ONLY_CAB).strip()
        cab_ids = [ONLY_CAB_STR] if ONLY_CAB_STR in cab_ids else []

    print("cabs:", len(cab_ids), "FIXED_EPSG:", fixed_epsg)

    if os.path.exists(OUT_ALL_JSONL):
        os.remove(OUT_ALL_JSONL)
    if SAVE_ALL_POINTS_CSV and os.path.exists(OUT_ALL_POINTS_CSV):
        os.remove(OUT_ALL_POINTS_CSV)

    n_ok, n_fail = 0, 0
    wrote_header = False

    with open(OUT_ALL_JSONL, "a", encoding="utf-8") as fout:
        for k, cab_id in enumerate(cab_ids, 1):
            print(f"[{k}/{len(cab_ids)}] cab={cab_id}")
            try:
                cab_out, points_df = process_one_cab(df, cab_id, edges_m, sidx, eid2ends, eid2uv, fixed_epsg, G_id)
                if cab_out is None:
                    continue

                # 每车 trips JSON（最终）
                out_path = os.path.join(OUT_DIR, f"cab_{cab_id}_trips.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(cab_out, f, ensure_ascii=False, indent=2)

                # 每车 points CSV（原始点匹配）
                if points_df is not None:
                    out_points = os.path.join(OUT_DIR, f"edge_hmm_cab{cab_id}_all_points.csv")
                    points_df.to_csv(out_points, index=False, encoding="utf-8-sig")

                    # 全体 points 合并
                    if SAVE_ALL_POINTS_CSV:
                        points_df.to_csv(
                            OUT_ALL_POINTS_CSV,
                            index=False,
                            encoding="utf-8-sig",
                            mode="a",
                            header=(not wrote_header)
                        )
                        wrote_header = True

                # 总 JSONL
                fout.write(json.dumps(cab_out, ensure_ascii=False) + "\n")

                n_ok += 1
            except Exception as e:
                n_fail += 1
                print("  FAIL:", repr(e))
                continue

    print("DONE. ok=", n_ok, "fail=", n_fail)
    print("ALL JSONL:", OUT_ALL_JSONL)
    if SAVE_ALL_POINTS_CSV:
        print("ALL points CSV:", OUT_ALL_POINTS_CSV)
    print("OUT_DIR:", OUT_DIR)


if __name__ == "__main__":
    main()
