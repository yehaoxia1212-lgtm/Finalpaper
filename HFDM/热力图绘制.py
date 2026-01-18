import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely import wkt
import matplotlib as mpl
import numpy as np
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap


def read_heat_csv(csv_path, crs="EPSG:4326"):
    df = pd.read_csv(csv_path)
    df["geometry"] = df["geometry"].apply(wkt.loads)
    return gpd.GeoDataFrame(df, geometry="geometry", crs=crs)


def compute_global_vmin_vmax(
    heat_root, schemes, cities, resolutions, time_map, num_sensors
):
    vals = []
    for scheme in schemes:
        for city in cities:
            for res in resolutions:
                for t in time_map.values():
                    csv_path = os.path.join(
                        heat_root, scheme,
                        f"{city}_{num_sensors}_grid{res}_time{t}.csv"
                    )
                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)
                        vals.append(df["score"])
    all_vals = pd.concat(vals, ignore_index=True)
    vmin = float(all_vals.min())
    vmax = float(all_vals.quantile(0.99))
    return vmin, vmax


# --- 不发黑的截断 YlOrRd：最大呈红色 ---
base_cmap = cm.get_cmap("YlOrRd")
cmap_trunc = LinearSegmentedColormap.from_list(
    "YlOrRd_trunc",
    base_cmap(np.linspace(0.0, 0.88, 256))
)


def plot_2x3_panel_toplabels(
    city,
    scheme,
    num_sensors,
    heat_root,
    time_map,
    resolutions,
    out_path,
    vmin,
    vmax,
    label_fontsize=11
):
    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(12, 8),
        constrained_layout=True
    )

    for i, res in enumerate(resolutions):
        for j, (time_label, t) in enumerate(time_map.items()):
            ax = axes[i, j]

            csv_path = os.path.join(
                heat_root, scheme,
                f"{city}_{num_sensors}_grid{res}_time{t}.csv"
            )
            gdf = read_heat_csv(csv_path)

            gdf.plot(
                column="score",
                ax=ax,
                cmap=cmap_trunc,
                vmin=vmin,
                vmax=vmax,
                edgecolor="white",
                linewidth=0.2,
                legend=False
            )

            # ✅ 标注放在“正上方”（轴外上边缘中间）
            ax.text(
                0.5, 1.02,
                f"{res}m  {time_label}  $N_s$={num_sensors}",
                transform=ax.transAxes,
                ha="center", va="bottom",
                fontsize=label_fontsize,
                color="black"
            )

            ax.set_axis_off()

    # 单一 colorbar
    sm = mpl.cm.ScalarMappable(
        cmap=cmap_trunc,
        norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    )
    sm._A = []
    cbar = fig.colorbar(
        sm,
        ax=axes,
        orientation="vertical",
        fraction=0.03,
        pad=0.02
    )
    cbar.set_label(r"$(N_{g,t})^{\beta}$", fontsize=11)

    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved panel: {out_path}")


if __name__ == "__main__":

    cities = ["chengdu", "manhattan", "san"]
    schemes = ["bus", "taxi", "joint"]
    resolutions = [500, 1000]

    time_map = {
        "08:00": 480,
        "14:00": 840,
        "18:00": 1080
    }

    num_sensors = 100

    heat_root = "./res_heat"
    out_root = "./figures/heat_panels"
    os.makedirs(out_root, exist_ok=True)

    vmin, vmax = compute_global_vmin_vmax(
        heat_root, schemes, cities, resolutions, time_map, num_sensors
    )
    print(f"Color scale: vmin={vmin:.6f}, vmax={vmax:.6f}")

    for scheme in schemes:
        for city in cities:
            out_path = os.path.join(
                out_root,
                f"{city}_{scheme}_Ns{num_sensors}_panel.png"
            )

            plot_2x3_panel_toplabels(
                city=city,
                scheme=scheme,
                num_sensors=num_sensors,
                heat_root=heat_root,
                time_map=time_map,
                resolutions=resolutions,
                out_path=out_path,
                vmin=vmin,
                vmax=vmax,
                label_fontsize=11
            )
