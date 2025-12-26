import json
import geopandas as gpd
import matplotlib.pyplot as plt
import os
# 1) 读入 GeoJSON（CRS84/经纬度可当 EPSG:4326 用来画图）
with open("manhattan_shape.json", "r", encoding="utf-8") as f:
    gj = json.load(f)

gdf = gpd.GeoDataFrame.from_features(gj["features"], crs="EPSG:4326")

# 2) MultiPolygon -> 拆成单个 Polygon（一块一行）
#    index_parts=False: 不把原索引拆成多级索引，后面更好处理
parts = gdf.explode(index_parts=False).reset_index(drop=True)
parts["poly_id"] = range(1, len(parts) + 1)

# 3) 画图 + 在每块上写编号
ax = parts.plot(edgecolor="black", facecolor="none", figsize=(10, 10))
for _, row in parts.iterrows():
    # representative_point() 比 centroid 更稳（凹多边形时标签不会跑到外面）
    x, y = row.geometry.representative_point().coords[0]
    ax.text(x, y, str(row["poly_id"]), ha="center", va="center", fontsize=8)

ax.set_title("Polygon parts with poly_id")
plt.axis("equal")
plt.show()

keep_id = 29  # <- 你把这里改成你要的编号
kept = parts.loc[parts["poly_id"] == keep_id].copy()

# 导出为新的 GeoJSON（只含这一块）
kept.to_file("manhattan_shape_cleaned.geojson", driver="GeoJSON", encoding="utf-8")
print("saved -> kept.geojson")

out_dir = "manhattan_shape_cleaned"
os.makedirs(out_dir, exist_ok=True)

# Shapefile 主文件名（会生成 .shp/.shx/.dbf/.prj 等多个文件）
shp_path = os.path.join(out_dir, "manhattan_shape_cleaned.shp")

kept.to_file(shp_path, driver="ESRI Shapefile", encoding="utf-8")