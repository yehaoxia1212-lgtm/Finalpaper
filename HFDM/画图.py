import os
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import MultipleLocator
from matplotlib import font_manager

city = 'chengdu'
resolution = 500

# ====== 字体：优先宋体，回退 ======
preferred_fonts = ['SimSun', 'Songti SC', 'STSong', 'Microsoft YaHei', 'SimHei']
available = {f.name for f in font_manager.fontManager.ttflist}
for f in preferred_fonts:
    if f in available:
        plt.rcParams['font.family'] = f
        break
plt.rcParams['axes.unicode_minus'] = False

# ====== 读取三类 score 结果 ======
bus = pd.read_csv(f'./res_bus/{city}/Score-bus_grid{resolution}.csv')
taxi = pd.read_csv(f'./res_taxi/{city}/Score-taxi_grid{resolution}.csv')
bus_taxi = pd.read_csv(f'./res_joint/{city}/Score-bus-taxi_grid{resolution}.csv')

# ====== 统一处理：0~200，且补 (0,0) ======
def prepare_df(df):
    df = df.copy()
    df['num_sensors'] = df['num_sensors'].astype(int)
    df = df[(df['num_sensors'] >= 0) & (df['num_sensors'] <= 200)]
    if 0 not in df['num_sensors'].values:
        df = pd.concat([pd.DataFrame({'num_sensors': [0], 'score': [0.0]}), df], ignore_index=True)
    return df.sort_values('num_sensors')

bus = prepare_df(bus)
taxi = prepare_df(taxi)
bus_taxi = prepare_df(bus_taxi)

# ====== 绘图 ======
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(bus['num_sensors'], bus['score'],
        label='公交', linestyle='--', marker='o', markersize=5, linewidth=2, zorder=4)

ax.plot(taxi['num_sensors'], taxi['score'],
        label='出租车', linestyle=':', marker='o', markersize=5, linewidth=2, zorder=5)

ax.plot(bus_taxi['num_sensors'], bus_taxi['score'],
        label='公交-出租车联合', linestyle='-', marker='o', markersize=6, linewidth=2.5,
        alpha=0.45, zorder=6)

# 坐标轴：0~200，且 y 从 0 开始
ax.set_xlim(0, 200)
y_max = max(bus['score'].max(), taxi['score'].max(), bus_taxi['score'].max())
ax.set_ylim(0, y_max * 1.05)

# 刻度
ax.xaxis.set_major_locator(MultipleLocator(40))
ax.yaxis.set_major_locator(MultipleLocator(20000))
ax.tick_params(axis='both', labelsize=14)

# 中文标签
# ax.set_xlabel('传感器数量', fontsize=16)
# ax.set_ylabel(r'监测效用 $\Phi_I$', fontsize=16)

ax.grid(True)
ax.legend(fontsize=12)

# 保存
os.makedirs('pic_result', exist_ok=True)
save_path = f'pic_result/高频动态监测效用对比_{city}_grid{resolution}_0-200.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Saved figure to: {save_path}")
