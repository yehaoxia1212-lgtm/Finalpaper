import osmnx as ox
import geopandas as gpd
import pandas as pd
import geopandas as gpd
import transbigdata
import os
def function01(geometry):
    x,y =geometry.exterior.coords.xy
    bottom_left_lon = x[0]
    bottom_left_lat = y[0]
    top_right_lon = x[2]
    top_right_lat = y[2]
    ser1 = pd.Series([bottom_left_lon,bottom_left_lat,top_right_lon,top_right_lat])

    return  ser1

def draw_grid():
    cq = gpd.read_file('./chengdu_roadmap/武侯区.json')
    # 画1000m网格
    result = transbigdata.area_to_grid(cq, accuracy=500, method='rect')
    # 保存到本地
    result[0].to_file('grid_wuhou.json', driver='GeoJSON')

    lon, lat = transbigdata.grid_to_centre([result[0].iloc[:, 0], result[0].iloc[:, 1]], result[1])
    output = result[0]
    output['center_lon'] = lon
    output['center_lat'] = lat
    output[['bottom_left_lon', 'bottom_left_lat', 'top_right_lon', 'top_right_lat']] = output['geometry'].apply(
        lambda x: function01(x))
    output['grid_id'] = [i for i in range(len(output))]
    output = output[
        ['LONCOL', 'LATCOL', 'center_lon', 'center_lat', 'grid_id', 'geometry', 'bottom_left_lon', 'bottom_left_lat',
         'top_right_lon', 'top_right_lat']]
    output.to_csv('grid_wuhou.csv')

    return output

def generate_grid(cq, RES, folder,city):
    # 创建输出文件夹
    os.makedirs(folder, exist_ok=True)

    # 生成网格
    result = transbigdata.area_to_grid(cq, accuracy=RES, method='rect')
    grid_gdf = result[0]
    grid_info = result[1]

    # 计算中心点
    lon, lat = transbigdata.grid_to_centre([grid_gdf.iloc[:, 0], grid_gdf.iloc[:, 1]], grid_info)
    grid_gdf['center_lon'] = lon
    grid_gdf['center_lat'] = lat

    # 计算左下角/右上角
    grid_gdf[['bottom_left_lon', 'bottom_left_lat', 'top_right_lon', 'top_right_lat']] = \
        grid_gdf['geometry'].apply(lambda x: function01(x))

    # 添加 grid_id
    grid_gdf['grid_id'] = range(len(grid_gdf))

    # 保存 GeoJSON
    geojson_path = os.path.join(folder, f'{city}_grid{RES}.geojson')
    grid_gdf.to_file(geojson_path, driver='GeoJSON')

    # 保存 CSV
    csv_path = os.path.join(folder, f'{city}_grid{RES}.csv')
    grid_gdf[['LONCOL', 'LATCOL', 'center_lon', 'center_lat', 'grid_id', 'geometry',
              'bottom_left_lon', 'bottom_left_lat', 'top_right_lon', 'top_right_lat']].to_csv(csv_path, index=False)

    # 保存 Shapefile 放在单独子文件夹
    shapefile_folder = os.path.join(folder, f'SHP_{RES}m')
    os.makedirs(shapefile_folder, exist_ok=True)
    shapefile_path = os.path.join(shapefile_folder, f'{city}_grid{RES}.shp')
    grid_gdf.to_file(shapefile_path, driver='ESRI Shapefile')

    print(f"完成：生成 GeoJSON, CSV, Shapefile -> {folder}")

if __name__ == '__main__':

# 设置完整城市名称和简写名称
#     place_name = "San Francisco County, California, USA"
#     city_name = "San_Francisco"
#
#     # 创建存储文件夹
#     os.makedirs(city_name, exist_ok=True)
#
#     # 仅下载边界 polygon
#     boundary_gdf = ox.geocode_to_gdf(place_name)
#     boundary_gdf.plot()
# #
# #     # 保存城市边界 GeoJSON
#     boundary_gdf.to_file(os.path.join(city_name, f"boundary_{city_name}.geojson"), driver="GeoJSON")


# 画网格
    city = 'chengdu'
    folder = f'./{city}_grid'
    cq = gpd.read_file(os.path.join(folder, 'chengdu_shape.geojson'))
    # 生成 500m 网格
    # generate_grid(cq, 3050, folder)
    generate_grid(cq, 500, folder,city)
    generate_grid(cq, 1000, folder,city)
