# -*- coding: utf-8 -*-
"""
GCJ-02 (Amap/Tencent) -> WGS84 conversion for taxi GPS points in CSV
Outputs:
  1) CSV (lon/lat/geometry updated to WGS84)
  2) GeoJSON (Point features, EPSG:4326), streamed writing via Fiona

Requirements: pandas, numpy, geopandas (installs fiona), shapely
"""

import os
import math
import numpy as np
import pandas as pd

import fiona
from fiona.crs import CRS as FionaCRS


# =========================
# 0) 配置：只改这里
# =========================
IN_CSV = r"D:\Finalpaper\LFDM\longquanyi\data_taxi\data_taxi_longquanyi_gcj_full.csv"
OUT_CSV = r"D:\Finalpaper\LFDM\longquanyi\data_taxi\data_taxi_longquanyi.csv"
OUT_GEOJSON = r"D:\Finalpaper\LFDM\longquanyi\data_taxi\data_taxi_longquanyi.geojson"

LON_COL = "lon"
LAT_COL = "lat"
GEOM_COL = "geometry"   # 若输入CSV没有该列，会自动创建

CHUNK_SIZE = 200_000    # 适合 81W 点

# 为了避免 Fiona schema/类型问题：统一将 properties 写为字符串（最稳）
ALL_PROPERTIES_AS_STR = True

# =========================
# 1) GCJ-02 -> WGS84（公开常用实现）
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
    return lon + dlon, lat + dlat

def gcj02_to_wgs84(lon, lat):
    # WGS = GCJ - (GCJ - (WGS->GCJ))
    if out_of_china(lon, lat):
        return lon, lat
    glon, glat = wgs84_to_gcj02(lon, lat)
    return lon * 2 - glon, lat * 2 - glat

def gcj02_to_wgs84_array(lon_arr, lat_arr):
    lon_arr = lon_arr.astype(float)
    lat_arr = lat_arr.astype(float)
    wlon = np.empty_like(lon_arr, dtype=float)
    wlat = np.empty_like(lat_arr, dtype=float)
    for i in range(len(lon_arr)):
        wlon[i], wlat[i] = gcj02_to_wgs84(float(lon_arr[i]), float(lat_arr[i]))
    return wlon, wlat

# =========================
# 2) geometry WKT（CSV里存WKT最通用）
# =========================
def make_point_wkt(lon_arr, lat_arr):
    lon_s = np.char.mod("%.10f", lon_arr)
    lat_s = np.char.mod("%.10f", lat_arr)
    return np.char.add(np.char.add("POINT (", np.char.add(lon_s, " ")), np.char.add(lat_s, ")"))

# =========================
# 3) Fiona schema（流式写 GeoJSON）
# =========================
def build_schema(columns, geom_col):
    props = {}
    for c in columns:
        if c == geom_col:
            continue
        props[c] = "str" if ALL_PROPERTIES_AS_STR else "str"
    # geometry列不作为properties写入
    return {"geometry": "Point", "properties": props}

def sanitize_props(d):
    out = {}
    for k, v in d.items():
        if v is None:
            out[k] = None
        elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            out[k] = None
        else:
            out[k] = v
    return out

# =========================
# 4) 主流程
# =========================
def convert_and_save(in_csv, out_csv, out_geojson, lon_col, lat_col, geom_col, chunksize):
    if not os.path.exists(in_csv):
        raise FileNotFoundError(in_csv)

    if os.path.abspath(in_csv) == os.path.abspath(out_csv):
        raise ValueError("输出CSV不能与输入CSV同名（避免覆盖风险）")

    if os.path.exists(out_csv):
        os.remove(out_csv)
    if os.path.exists(out_geojson):
        os.remove(out_geojson)

    reader = pd.read_csv(in_csv, chunksize=chunksize)

    first = True
    total = 0
    geo_writer = None

    for chunk in reader:
        if lon_col not in chunk.columns or lat_col not in chunk.columns:
            raise ValueError(f"CSV缺少列：{lon_col}/{lat_col}")

        # 如果没有geometry列，创建（最终会写WKT）
        if geom_col not in chunk.columns:
            chunk[geom_col] = None

        # 计算WGS lon/lat，并覆盖原列（列名不变）
        lon = chunk[lon_col].to_numpy(dtype=float)
        lat = chunk[lat_col].to_numpy(dtype=float)
        wlon, wlat = gcj02_to_wgs84_array(lon, lat)

        chunk[lon_col] = wlon
        chunk[lat_col] = wlat
        chunk[geom_col] = make_point_wkt(wlon, wlat)

        # 写CSV（只保留“新列值”，列名不变）
        chunk.to_csv(out_csv, index=False, mode="w" if first else "a", header=first, encoding="utf-8-sig")

        # 初始化GeoJSON writer
        if first:
            schema = build_schema(chunk.columns, geom_col)
            geo_writer = fiona.open(
                out_geojson,
                mode="w",
                driver="GeoJSON",
                schema=schema,
                crs=FionaCRS.from_epsg(4326),
                encoding="utf-8"
            )

        # 写GeoJSON（geometry来自 wlon/wlat；properties为除geometry外所有列）
        prop_cols = [c for c in chunk.columns if c != geom_col]
        props_df = chunk[prop_cols].where(pd.notnull(chunk[prop_cols]), None)

        if ALL_PROPERTIES_AS_STR:
            for c in props_df.columns:
                props_df[c] = props_df[c].apply(lambda x: None if x is None else str(x))

        for i in range(len(chunk)):
            x = float(wlon[i])
            y = float(wlat[i])
            if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
                continue

            props = sanitize_props(props_df.iloc[i].to_dict())
            feat = {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": (x, y)},
                "properties": props
            }
            geo_writer.write(feat)

        total += len(chunk)
        print(f"Processed: {total} rows")
        first = False

    if geo_writer is not None:
        geo_writer.close()

    print("\nDone.")
    print("CSV saved    :", out_csv)
    print("GeoJSON saved :", out_geojson)
    print("Total rows   :", total)


if __name__ == "__main__":
    convert_and_save(
        in_csv=IN_CSV,
        out_csv=OUT_CSV,
        out_geojson=OUT_GEOJSON,
        lon_col=LON_COL,
        lat_col=LAT_COL,
        geom_col=GEOM_COL,
        chunksize=CHUNK_SIZE
    )