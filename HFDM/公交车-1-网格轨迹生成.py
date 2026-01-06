import geopandas as gpd
import pandas as pd

def add_pass_grid_id(data_bus, data_grid, route_col="route_id", grid_col="grid_id"):
    """
    给 data_bus 每条线路添加经过的网格序列 pass_grid_id（按线路顺序排列）

    返回：一个新的 GeoDataFrame = 原 data_bus + 一列 pass_grid_id（list）
    """

    # ---------- 1. 为 sjoin 准备轻量副本，不改原 data_bus / data_grid ----------
    bus = data_bus[[route_col, "geometry"]].copy()
    grid = data_grid[[grid_col, "geometry"]].copy()

    # ---------- 2. CRS 对齐（只改副本，不动原始） ----------
    if bus.crs is None and grid.crs is not None:
        bus = bus.set_crs(grid.crs)
    elif grid.crs is None and bus.crs is not None:
        grid = grid.set_crs(bus.crs)
    elif (bus.crs is not None) and (grid.crs is not None) and (bus.crs != grid.crs):
        grid = grid.to_crs(bus.crs)

    # ---------- 3. 空间连接：哪条线路经过哪些网格 ----------
    joined = gpd.sjoin(
        bus,
        grid,
        how="inner",
        predicate="intersects"
    )
    # joined：包含 route_col, geometry(线路), grid_col, index 等

    # 补上 polygon 的 geometry，叫 grid_geom，方便后面算顺序
    joined = joined.merge(
        grid[[grid_col, "geometry"]].rename(columns={"geometry": "grid_geom"}),
        on=grid_col,
        how="left"
    )

    # ---------- 4. 计算该 grid 在线路上的“顺序位置” ----------
    def _pos_on_line(row):
        line = row["geometry"]      # LineString
        poly = row["grid_geom"]     # Polygon

        inter = line.intersection(poly)
        if inter.is_empty:
            pt = poly.representative_point()
        else:
            pt = inter.representative_point()

        return line.project(pt)

    joined["pos_on_line"] = joined.apply(_pos_on_line, axis=1)

    # ---------- 5. 按线路顺序排序 + 去掉重复 grid ----------
    joined_sorted = (
        joined
        .sort_values([route_col, "pos_on_line"])
        .drop_duplicates([route_col, grid_col], keep="first")
    )

    # ---------- 6. 聚合成每条线路的 grid_id 列表 ----------
    grid_seq = (
        joined_sorted
        .groupby(route_col)[grid_col]
        .apply(list)
        .reset_index(name="pass_grid_id")
    )

    # （可选）确保列表里是基础类型（int/str），方便写 GeoJSON
    grid_seq["pass_grid_id"] = grid_seq["pass_grid_id"].apply(
        lambda lst: [int(x) for x in lst]  # 如果 grid_id 本来不是 int 可以自己改
    )

    # ---------- 7. 合回原 data_bus：只多一列，其他一列不少 ----------
    data_bus_out = data_bus.copy()
    data_bus_out = data_bus_out.merge(grid_seq, on=route_col, how="left")

    return data_bus_out

city = 'chengdu'
resolution = 1000

data_bus = gpd.read_file(f'./data_bus/{city}/data_bus_{city}.geojson')
data_grid = gpd.read_file(f'./data_grid/{city}_grid/{city}_grid{resolution}.geojson')

data_bus_with_grid = add_pass_grid_id(data_bus, data_grid)
data_bus_with_grid.to_csv(f'./data_bus/{city}/data_bus_{city}_grid{resolution}.csv', index=False)
data_bus_with_grid.to_file(f'./data_bus/{city}/data_bus_{city}_grid{resolution}.geojson', driver='GeoJSON')
