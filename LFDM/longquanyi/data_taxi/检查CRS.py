# -*- coding: utf-8 -*-
"""
检查成都轨迹点坐标体系：WGS84 / GCJ-02 / BD-09
思路：同一批点分别按三种假设转换到WGS84，与OSM路网(假定WGS84)计算“点到最近路段距离(米)”。
哪一种假设的距离中位数最小 => 最可能是该坐标体系。
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import nearest_points
from pyproj import CRS

# =========================
# 0) 配置：改这里就行
# =========================
CSV_PATH = r"/LFDM/longquanyi/data_taxi/data_taxi_longquanyi.csv"  # TODO: 改成你的 taxi CSV
EDGES_PATH = r"/LFDM/longquanyi/roadmap_taxi_longquanyi/edge.geojson"  # TODO: 改成你的 edge.geojson（也可相对路径）

LON_COL = "lon"
LAT_COL = "lat"
CAB_COL = "cab_id"
TIME_COL = "date_time"

# 抽样参数：用于诊断，不必用全量81万点
SAMPLE_CABS = 50        # 抽多少辆车
PER_CAB_N = 300         # 每辆车抽多少点
DIST_SAMPLE_N = 20000   # 最终用于算距离的点数（上限）
MAX_SNAP_DIST_M = None  # 这里只做诊断不筛距离；你也可设成 80/50/30 来看敏感性
RANDOM_STATE = 0

# =========================
# 1) GCJ-02 / BD-09 / WGS84 转换（常用公开实现）
# =========================
PI = np.pi
A = 6378245.0
EE = 0.00669342162296594323

def out_of_china(lon, lat):
    return not (73.66 < lon < 135.05 and 3.86 < lat < 53.55)

def _transform_lat(lon, lat):
    ret = -100.0 + 2.0*lon + 3.0*lat + 0.2*lat*lat + 0.1*lon*lat + 0.2*np.sqrt(abs(lon))
    ret += (20.0*np.sin(6.0*lon*PI) + 20.0*np.sin(2.0*lon*PI)) * 2.0/3.0
    ret += (20.0*np.sin(lat*PI) + 40.0*np.sin(lat/3.0*PI)) * 2.0/3.0
    ret += (160.0*np.sin(lat/12.0*PI) + 320*np.sin(lat*PI/30.0)) * 2.0/3.0
    return ret

def _transform_lon(lon, lat):
    ret = 300.0 + lon + 2.0*lat + 0.1*lon*lon + 0.1*lon*lat + 0.1*np.sqrt(abs(lon))
    ret += (20.0*np.sin(6.0*lon*PI) + 20.0*np.sin(2.0*lon*PI)) * 2.0/3.0
    ret += (20.0*np.sin(lon*PI) + 40.0*np.sin(lon/3.0*PI)) * 2.0/3.0
    ret += (150.0*np.sin(lon/12.0*PI) + 300.0*np.sin(lon/30.0*PI)) * 2.0/3.0
    return ret

def wgs84_to_gcj02(lon, lat):
    if out_of_china(lon, lat):
        return lon, lat
    dlat = _transform_lat(lon - 105.0, lat - 35.0)
    dlon = _transform_lon(lon - 105.0, lat - 35.0)
    radlat = lat / 180.0 * PI
    magic = np.sin(radlat)
    magic = 1 - EE * magic * magic
    sqrtmagic = np.sqrt(magic)
    dlat = (dlat * 180.0) / ((A * (1 - EE)) / (magic * sqrtmagic) * PI)
    dlon = (dlon * 180.0) / (A / sqrtmagic * np.cos(radlat) * PI)
    mg_lat = lat + dlat
    mg_lon = lon + dlon
    return mg_lon, mg_lat

def gcj02_to_wgs84(lon, lat):
    if out_of_china(lon, lat):
        return lon, lat
    glon, glat = wgs84_to_gcj02(lon, lat)
    return lon * 2 - glon, lat * 2 - glat

def bd09_to_gcj02(bd_lon, bd_lat):
    x = bd_lon - 0.0065
    y = bd_lat - 0.006
    z = np.sqrt(x * x + y * y) - 0.00002 * np.sin(y * PI)
    theta = np.arctan2(y, x) - 0.000003 * np.cos(x * PI)
    gg_lon = z * np.cos(theta)
    gg_lat = z * np.sin(theta)
    return gg_lon, gg_lat

def bd09_to_wgs84(bd_lon, bd_lat):
    glon, glat = bd09_to_gcj02(bd_lon, bd_lat)
    return gcj02_to_wgs84(glon, glat)

# =========================
# 2) 投影到局部 UTM（用米做距离）
# =========================
def to_local_utm(gdf):
    if gdf.crs is None:
        raise ValueError("GeoDataFrame 缺少 CRS，请先 set_crs('EPSG:4326', allow_override=True)")
    if CRS(gdf.crs).is_projected:
        return gdf.copy()
    lon, lat = gdf.geometry.unary_union.centroid.x, gdf.geometry.unary_union.centroid.y
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return gdf.to_crs(epsg=epsg)

# =========================
# 3) 兼容版 nearest：拿到最近 edge 的 iloc（int）
# =========================
def _nearest_edge_iloc(sidx, geom):
    res = sidx.nearest(geom)
    # 生成器：老版本
    if hasattr(res, "__iter__") and not isinstance(res, (np.ndarray, list, tuple)):
        return int(next(iter(res)))

    arr = np.asarray(res)
    # 一维：[edge_idx]
    if arr.ndim == 1:
        return int(arr[0])
    # 二维：[[pt_idx],[edge_idx]]
    if arr.ndim == 2:
        return int(arr[1, 0])

    return int(list(res)[0])

# =========================
# 4) 计算到最近路段距离（米）
# =========================
def nearest_road_distances_m(points_m, edges_m, sample_n=20000, random_state=0, max_snap_dist_m=None):
    pts = points_m.copy()
    if sample_n is not None and len(pts) > sample_n:
        pts = pts.sample(sample_n, random_state=random_state)

    sidx = edges_m.sindex
    dists = []

    for p in pts.geometry:
        if p is None or p.is_empty:
            continue
        edge_iloc = _nearest_edge_iloc(sidx, p)
        e = edges_m.geometry.iloc[edge_iloc]  # shapely LineString

        q = nearest_points(p, e)[1]
        dist = float(p.distance(q))

        if (max_snap_dist_m is None) or (dist <= max_snap_dist_m):
            dists.append(dist)

    return np.asarray(dists, dtype=float)

def dist_stats(arr):
    if len(arr) == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "p90": np.nan, "p95": np.nan}
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
    }

# =========================
# 5) 主诊断流程
# =========================
def diagnose_taxi_coord_system():
    # ---- 读 taxi ----
    df = pd.read_csv(CSV_PATH)

    # 基础范围检查：你这一步已经证明是“度”
    print("lon range:", df[LON_COL].min(), df[LON_COL].max())
    print("lat range:", df[LAT_COL].min(), df[LAT_COL].max())

    # 抽车、抽点
    rng = np.random.default_rng(RANDOM_STATE)
    cab_ids = df[CAB_COL].dropna().unique()
    if len(cab_ids) == 0:
        raise ValueError("cab_id 列为空或不存在")

    chosen = rng.choice(cab_ids, size=min(SAMPLE_CABS, len(cab_ids)), replace=False)
    sub_list = []
    for cab in chosen:
        sub = df[df[CAB_COL] == cab]
        if len(sub) > PER_CAB_N:
            sub = sub.sample(PER_CAB_N, random_state=RANDOM_STATE)
        sub_list.append(sub)
    subdf = pd.concat(sub_list, ignore_index=True)

    # ---- 读 edges（OSM 路网，按你说的当作 WGS84 / EPSG:4326）----
    edges = gpd.read_file(EDGES_PATH)[["edge_id", "geometry"]]
    if edges.crs is None:
        edges = edges.set_crs("EPSG:4326", allow_override=True)
    else:
        # 如果不是4326，统一到4326
        if str(edges.crs).upper() != "EPSG:4326":
            edges = edges.to_crs("EPSG:4326")

    # ---- 构造三套点：WGS / GCJ->WGS / BD->WGS ----
    # A) 原始当 WGS
    gdf_wgs = gpd.GeoDataFrame(
        subdf.copy(),
        geometry=[Point(xy) for xy in zip(subdf[LON_COL], subdf[LAT_COL])],
        crs="EPSG:4326"
    )

    # B) 原始当 GCJ
    lon0 = subdf[LON_COL].to_numpy(dtype=float)
    lat0 = subdf[LAT_COL].to_numpy(dtype=float)
    lon_gcj2w = np.empty_like(lon0)
    lat_gcj2w = np.empty_like(lat0)
    for i in range(len(lon0)):
        lon_gcj2w[i], lat_gcj2w[i] = gcj02_to_wgs84(lon0[i], lat0[i])

    gdf_gcj_as_wgs = gpd.GeoDataFrame(
        subdf.copy(),
        geometry=[Point(xy) for xy in zip(lon_gcj2w, lat_gcj2w)],
        crs="EPSG:4326"
    )

    # C) 原始当 BD
    lon_bd2w = np.empty_like(lon0)
    lat_bd2w = np.empty_like(lat0)
    for i in range(len(lon0)):
        lon_bd2w[i], lat_bd2w[i] = bd09_to_wgs84(lon0[i], lat0[i])

    gdf_bd_as_wgs = gpd.GeoDataFrame(
        subdf.copy(),
        geometry=[Point(xy) for xy in zip(lon_bd2w, lat_bd2w)],
        crs="EPSG:4326"
    )

    # ---- 投影到米制并计算距离 ----
    edges_m = to_local_utm(edges)
    wgs_m   = to_local_utm(gdf_wgs)
    gcj_m   = to_local_utm(gdf_gcj_as_wgs)
    bd_m    = to_local_utm(gdf_bd_as_wgs)

    d_wgs = nearest_road_distances_m(wgs_m, edges_m, sample_n=DIST_SAMPLE_N,
                                     random_state=RANDOM_STATE, max_snap_dist_m=MAX_SNAP_DIST_M)
    d_gcj = nearest_road_distances_m(gcj_m, edges_m, sample_n=DIST_SAMPLE_N,
                                     random_state=RANDOM_STATE, max_snap_dist_m=MAX_SNAP_DIST_M)
    d_bd  = nearest_road_distances_m(bd_m,  edges_m, sample_n=DIST_SAMPLE_N,
                                     random_state=RANDOM_STATE, max_snap_dist_m=MAX_SNAP_DIST_M)

    s_wgs = dist_stats(d_wgs)
    s_gcj = dist_stats(d_gcj)
    s_bd  = dist_stats(d_bd)

    print("\n=== Distance to nearest road (meters) ===")
    print("Assume RAW is WGS84     :", s_wgs)
    print("Assume RAW is GCJ02->WGS:", s_gcj)
    print("Assume RAW is BD09 ->WGS:", s_bd)

    medians = {
        "RAW=WGS84": s_wgs["median"],
        "RAW=GCJ02 (convert->WGS)": s_gcj["median"],
        "RAW=BD09  (convert->WGS)": s_bd["median"],
    }
    best = min(medians, key=medians.get)
    print("\n>>> Best (smallest median distance):", best, "median=", medians[best])

if __name__ == "__main__":
    diagnose_taxi_coord_system()
