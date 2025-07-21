import pandas as pd
import matplotlib.pyplot as plt

# 데이터 파일 경로 설정
data_path = './dataFile/'
map_file = f'{data_path}area_map.csv'
struct_file = f'{data_path}area_struct.csv'
category_file = f'{data_path}area_category.csv'

# CSV 파일 불러오기 (UTF-8-SIG 인코딩 사용)
df_map = pd.read_csv(map_file, encoding='utf-8-sig')
df_struct = pd.read_csv(struct_file, encoding='utf-8-sig')
df_category = pd.read_csv(category_file, encoding='utf-8-sig')

# 열 이름과 데이터 값의 공백 제거
df_category.columns = df_category.columns.str.strip()
df_category['struct'] = df_category['struct'].str.strip()

# 데이터 병합 및 area 1 필터링
df_struct = pd.merge(df_struct, df_category, on='category', how='left')
df_merged = pd.merge(df_map, df_struct, on=['x', 'y'], how='left')
df_merged['struct'] = df_merged['struct'].fillna('None')

# 시각화
plt.figure(figsize=(10, 12))

# 그리드 라인
plt.grid(True)

# 구조물 시각화
for idx, row in df_merged.iterrows():
    if row['struct'] == 'Apartment' or row['struct'] == 'Building':
        plt.scatter(row['x'], row['y'], c='brown', marker='o')
    elif row['struct'] == 'BandalgomCoffee':
        plt.scatter(row['x'], row['y'], c='green', marker='s')
    elif row['struct'] == 'MyHome':
        plt.scatter(row['x'], row['y'], c='green', marker='^')

# 건설 현장 시각화
construction_sites = df_merged[df_merged['ConstructionSite'] == 1]
for idx, row in construction_sites.iterrows():
    plt.scatter(row['x'], row['y'], c='grey', marker='s', s=200)

# 축 설정
plt.xlim(0, 16)
plt.ylim(0, 16)
plt.gca().invert_yaxis() # Y축을 위에서 아래로
plt.gca().set_aspect('equal', adjustable='box')

# (보너스) 범례
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Apartment/Building', markerfacecolor='brown', markersize=10),
    plt.Line2D([0], [0], marker='s', color='w', label='BandalgomCoffee', markerfacecolor='green', markersize=10),
    plt.Line2D([0], [0], marker='^', color='w', label='MyHome', markerfacecolor='green', markersize=10),
    plt.Line2D([0], [0], marker='s', color='w', label='Construction Site', markerfacecolor='grey', markersize=10, alpha=0.5)
]
plt.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2), mode="expand", borderaxespad=0)

# 이미지 파일 저장
plt.savefig('outputFiles/map.png')

print('outputFiles/map.png 파일이 저장되었습니다.')
