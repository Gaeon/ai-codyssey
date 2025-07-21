
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# 데이터 파일 경로 설정
data_path = '../dataFile/'
struct_file = f'{data_path}area_struct.csv'
category_file = f'{data_path}area_category.csv'

# CSV 파일 불러오기
df_struct = pd.read_csv(struct_file)
df_category = pd.read_csv(category_file)

# 열 이름과 데이터 값의 공백 제거 (area_category.csv)
df_category.columns = df_category.columns.str.strip()
df_category['struct'] = df_category['struct'].str.strip()

# 데이터 병합
df_merged = pd.merge(df_struct, df_category, on='category', how='left')

# 15x15 그리드 생성
fig, ax = plt.subplots(figsize=(10, 12))  # Increased figure height for legends

# area별 색상 지정 및 범례 핸들 생성
area_colors = {
    0: 'lightgrey',
    1: 'lightblue',
    2: 'lightgreen',
    3: 'lightyellow'
}

area_legend_patches = []
for area_val, color in area_colors.items():
    area_legend_patches.append(mpatches.Patch(
        color=color, label=f'Area {area_val}'))

# category별 마커 및 색상 지정
category_markers = {
    1: {'marker': 's', 'color': 'brown', 'label': 'Apartment'},
    2: {'marker': 'o', 'color': 'darkblue', 'label': 'Building'},
    3: {'marker': '^', 'color': 'green', 'label': 'MyHome'},
    4: {'marker': 'D', 'color': 'purple', 'label': 'BandalgomCoffee'}
}

category_legend_elements = []
for cat_val, props in category_markers.items():
    category_legend_elements.append(plt.Line2D([0], [0],
                                               marker=props['marker'],
                                               color='w',
                                               label=props['label'],
                                               markerfacecolor=props['color'],
                                               markersize=10))

# 각 셀에 area 색상 채우기
for index, row in df_merged.iterrows():
    x, y, area = row['x'], row['y'], row['area']
    ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1,
                 color=area_colors.get(area, 'white')))

# 각 셀에 category 마커 표시
for index, row in df_merged.iterrows():
    x, y, category = row['x'], row['y'], row['category']
    if category in category_markers:
        props = category_markers[category]
        # zorder로 마커가 위에 오도록
        ax.scatter(x, y, marker=props['marker'],
                   color=props['color'], s=100, zorder=2)

# 축 설정
ax.set_xlim(0.5, 15.5)
ax.set_ylim(0.5, 15.5)
ax.set_xticks(range(1, 16))
ax.set_yticks(range(1, 16))
ax.set_aspect('equal', adjustable='box')
ax.invert_yaxis()
plt.grid(True)

# 범례 추가
# 첫 번째 범례 (Area)
legend1 = ax.legend(handles=area_legend_patches, loc='lower left',
                     bbox_to_anchor=(0, 1.02, 0.5, 0.2), mode="expand",
                     borderaxespad=0, title='Area Colors')
ax.add_artist(legend1)  # 첫 번째 범례를 추가하고, 두 번째 범례를 위해 ax에 다시 추가

# 두 번째 범례 (Category)
legend2 = ax.legend(handles=category_legend_elements, loc='lower right',
                     bbox_to_anchor=(0.5, 1.02, 0.5, 0.2), mode="expand",
                     borderaxespad=0,
                     title='Category Markers')

# 이미지 파일 저장
output_path = 'area_category_map.png'
plt.savefig(output_path)

print(f'{output_path} 파일이 저장되었습니다.')
