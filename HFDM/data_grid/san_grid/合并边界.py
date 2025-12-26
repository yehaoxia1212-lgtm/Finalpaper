import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon
import os

# 1️⃣ 读取 GeoJSON
gdf = gpd.read_file('Current_Supervisor_Districts_20251127.geojson')

# 2️⃣ 保留 Vacant 第0条
vacant_idx = gdf[gdf['sup_name'] == 'Vacant'].index[0]
geom = gdf.loc[vacant_idx, 'geometry']
if isinstance(geom, MultiPolygon):
    gdf.loc[vacant_idx, 'geometry'] = list(geom.geoms)[0]  # 第0条
# 如果已经是 Polygon，则无需操作

# 3️⃣ 保留 Matt Dorsey 第3条
matt_idx = gdf[gdf['sup_name'] == 'Matt Dorsey'].index[0]  # 如果有多行 Matt，可以调整索引
geom = gdf.loc[matt_idx, 'geometry']
if isinstance(geom, MultiPolygon):
    gdf.loc[matt_idx, 'geometry'] = list(geom.geoms)[3]  # 第3条

# 4️⃣ 其他行政区保留全部 geometry
# 如果需要删除不需要的行，可以在这里过滤

# 5️⃣ 合并所有 geometry
merged_geom = unary_union(gdf.geometry)

# 6️⃣ 创建单行 GeoDataFrame
merged_gdf = gpd.GeoDataFrame([1], geometry=[merged_geom], columns=['id'], crs=gdf.crs)

# 7️⃣ 输出文件夹
out_folder = './'
os.makedirs(out_folder, exist_ok=True)

# 8️⃣ 保存文件
merged_gdf.to_file(os.path.join(out_folder, 'merged_boundary.geojson'), driver='GeoJSON')
# merged_gdf.to_file(os.path.join(out_folder, 'merged_boundary.shp'), driver='ESRI Shapefile')
merged_gdf['geometry_wkt'] = merged_gdf.geometry.apply(lambda x: x.wkt)
merged_gdf.to_csv(os.path.join(out_folder, 'merged_boundary.csv'), index=False)

print(f"完成：边界已融合并保存到 {out_folder}")
