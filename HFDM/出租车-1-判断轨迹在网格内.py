import geopandas as gpd

def add_grid_id(data_taxi, data_grid):
    """
    data_taxi: GeoDataFrame，POINT
    data_grid: GeoDataFrame，POLYGON，必须带 grid_id
    返回：在 data_taxi 中新增 grid_id 列
    """
    # 确保 CRS 一致
    if data_taxi.crs != data_grid.crs:
        data_grid = data_grid.to_crs(data_taxi.crs)

    # 仅保留 grid_id + geometry
    grid_gdf = data_grid[["grid_id", "geometry"]]

    # 空间连接（点-网格）
    joined = gpd.sjoin(
        data_taxi,
        grid_gdf,
        how="left",
        predicate="intersects"   # 点在边界时也能匹配
    )

    # 只返回原列 + 新增的 grid_id
    data_taxi["grid_id"] = joined["grid_id"].values

    return data_taxi

if __name__ == '__main__':
    cities = ['san','manhattan','chengdu']
    res = [1000,500]

    for city in cities:
        for resolution in res:
            data_grid = gpd.read_file(f'./data_grid/{city}_grid/{city}_grid{resolution}.geojson')
            data_taxi = gpd.read_file(f'./data_taxi/{city}/data_taxi_{city}.geojson')
            data_taxi = add_grid_id(data_taxi, data_grid)
            data_taxi.to_file(f'./data_taxi/{city}/data_taxi_{city}_grid{resolution}.geojson',driver='GeoJSON')
            data_taxi.to_csv(f'./data_taxi/{city}/data_taxi_{city}_grid{resolution}.csv',index=False)
        print(city)