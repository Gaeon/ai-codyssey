import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

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

# 데이터 병합
df_struct = pd.merge(df_struct, df_category, on='category', how='left')
df_merged = pd.merge(df_map, df_struct, on=['x', 'y'], how='left')
df_merged['struct'] = df_merged['struct'].fillna('None')

# 시작점 설정
start_node = (df_merged[df_merged['struct'] == 'MyHome']['x'].iloc[0], 
              df_merged[df_merged['struct'] == 'MyHome']['y'].iloc[0])
obstacles = set(zip(df_merged[df_merged['ConstructionSite'] == 1]['x'], df_merged[df_merged['ConstructionSite'] == 1]['y']))

# BandalgomCoffee 여러 개일 경우 모두 고려하여 가장 가까운 지점 선택
cafe_nodes = list(zip(df_merged[df_merged['struct'] == 'BandalgomCoffee']['x'],
                      df_merged[df_merged['struct'] == 'BandalgomCoffee']['y']))

# BFS 알고리즘 구현
def bfs(start, end, obstacles):
    queue = deque([[start]])
    visited = {start}

    while queue:
        path = queue.popleft()
        x, y = path[-1]

        if (x, y) == end:
            return path

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy

            if 1 <= nx <= 15 and 1 <= ny <= 15 and (nx, ny) not in visited and (nx, ny) not in obstacles:
                visited.add((nx, ny))
                new_path = list(path)
                new_path.append((nx, ny))
                queue.append(new_path)

    return None

# 최단 경로 초기화
shortest = None
min_length = float('inf')

# 각 반달곰 커피 지점에 대해 BFS 수행
for end_node in cafe_nodes:
    path = bfs(start_node, end_node, obstacles)
    if path and len(path) < min_length:
        shortest = path
        min_length = len(path)

# 최종 최단 경로 설정
shortest_path = shortest

# 경로를 CSV 파일로 저장
if shortest_path:
    path_df = pd.DataFrame(shortest_path, columns=['x', 'y'])
    path_df.to_csv('outputFiles/home_to_cafe.csv', index=False)
    print('outputFiles/home_to_cafe.csv 파일이 저장되었습니다.')

    # 시각화
    plt.figure(figsize=(10, 12))
    plt.grid(True)

    for idx, row in df_merged.iterrows():
        if row['struct'] == 'Apartment' or row['struct'] == 'Building':
            plt.scatter(row['x'], row['y'], c='brown', marker='o')
        elif row['struct'] == 'BandalgomCoffee':
            plt.scatter(row['x'], row['y'], c='green', marker='s')
        elif row['struct'] == 'MyHome':
            plt.scatter(row['x'], row['y'], c='green', marker='^')

    construction_sites = df_merged[df_merged['ConstructionSite'] == 1]
    for idx, row in construction_sites.iterrows():
        plt.scatter(row['x'], row['y'], c='grey', marker='s', s=200)

    path_x, path_y = zip(*shortest_path)
    plt.plot(path_x, path_y, c='red', linewidth=2)

    plt.xlim(0, 16)
    plt.ylim(0, 16)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Apartment/Building', markerfacecolor='brown', markersize=10),
        plt.Line2D([0], [0], marker='s', color='w', label='BandalgomCoffee', markerfacecolor='green', markersize=10),
        plt.Line2D([0], [0], marker='^', color='w', label='MyHome', markerfacecolor='green', markersize=10),
        plt.Line2D([0], [0], marker='s', color='w', label='Construction Site', markerfacecolor='grey', markersize=10, alpha=0.5),
        plt.Line2D([0], [0], color='red', lw=2, label='Shortest Path')
    ]
    plt.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2), mode="expand", borderaxespad=0)

    plt.savefig('outputFiles/map_final.png')
    print('outputFiles/map_final.png 파일이 저장되었습니다.')
else:
    print('경로를 찾을 수 없습니다.')
