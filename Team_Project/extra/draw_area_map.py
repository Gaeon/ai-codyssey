import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 데이터 파일 경로 설정
data_path = '../dataFile/'
struct_file = f'{data_path}area_struct.csv'

# CSV 파일 불러오기
df_struct = pd.read_csv(struct_file)

# 15x15 그리드 생성
fig, ax = plt.subplots(figsize=(10, 10))

# area별 색상 지정 및 범례 핸들 생성
colors = {
    0: 'lightgrey',
    1: 'lightblue',
    2: 'lightgreen',
    3: 'lightyellow'
}

legend_patches = []
for area_val, color in colors.items():
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=color,
                 visible=False))  # 범례를 위한 더미 패치
    legend_patches.append(mpatches.Patch(
        color=color, label=f'Area {area_val}'))

# 각 셀에 색상 채우기
for index, row in df_struct.iterrows():
    x, y, area = row['x'], row['y'], row['area']
    ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1,
                 1, color=colors.get(area, 'white')))

# 축 설정
ax.set_xlim(0.5, 15.5)
ax.set_ylim(0.5, 15.5)
ax.set_xticks(range(1, 16))
ax.set_yticks(range(1, 16))
ax.set_aspect('equal', adjustable='box')
ax.invert_yaxis()
plt.grid(True)

# 범례 추가
ax.legend(handles=legend_patches, loc='upper right', title='Area Colors')

# 이미지 파일 저장
output_path = 'area_map.png'
plt.savefig(output_path)

print(f'{output_path} 파일이 저장되었습니다.')