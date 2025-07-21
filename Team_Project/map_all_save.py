import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from collections import deque

# --- Data Loading ---
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
df_struct_with_category = pd.merge(
    df_struct, df_category, on='category', how='left')
df_merged = pd.merge(df_map, df_struct_with_category,
                     on=['x', 'y'], how='left')
df_merged['struct'] = df_merged['struct'].fillna('None')

# --- BFS Algorithm ---


def bfs(start, end, obstacles, grid_size=15):
    queue = deque([[start]])
    visited = {start}

    if start in obstacles or end in obstacles:
        return None

    while queue:
        path = queue.popleft()
        x, y = path[-1]

        if (x, y) == end:
            return path

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy

            if (1 <= nx <= grid_size and 1 <= ny <= grid_size and
                    (nx, ny) not in visited and (nx, ny) not in obstacles):
                visited.add((nx, ny))
                new_path = list(path)
                new_path.append((nx, ny))
                queue.append(new_path)
    return None

# --- Visualization Helper Function ---


def plot_map(
    df_data,
    path_coords,
    title,
    output_image_path,
    extra_legend_elements=None,
    start_node=None,
    visited_structures=None
):
    plt.figure(figsize=(10, 12))  # Increased figure height for legends
    plt.grid(True)

    # Plot all cells with area colors (background)
    area_colors = {
        0: 'lightgrey',
        1: 'lightblue',
        2: 'lightgreen',
        3: 'lightyellow'
    }
    for index, row in df_data.iterrows():
        x, y, area = row['x'], row['y'], row['area']
        plt.gca().add_patch(
            plt.Rectangle(
                (x - 0.5, y - 0.5),
                1,
                1,
                color=area_colors.get(area, 'white')
            )
        )

    # Plot structures and construction sites
    for idx, row in df_data.iterrows():
        if row['struct'] == 'Apartment' or row['struct'] == 'Building':
            plt.scatter(row['x'], row['y'], c='brown', marker='o', zorder=3)
        elif row['struct'] == 'BandalgomCoffee':
            plt.scatter(row['x'], row['y'], c='green', marker='s', zorder=3)
        elif row['struct'] == 'MyHome':
            plt.scatter(row['x'], row['y'], c='green', marker='^', zorder=3)

    construction_sites = df_data[df_data['ConstructionSite'] == 1]
    for idx, row in construction_sites.iterrows():
        plt.scatter(
            row['x'], row['y'],
            c='grey',
            marker='s',
            s=200,
            zorder=4
        )  # Construction sites on top

    # Plot the path
    if path_coords:
        path_x, path_y = zip(*path_coords)
        # Path below structures but above background
        plt.plot(path_x, path_y, c='red', linewidth=2, zorder=2)

    # Plot start and visited structures for bonus task
    if start_node:
        plt.scatter(start_node[0], start_node[1], c='blue',
                    marker='*', s=300, label='Start (MyHome)', zorder=5)
    if visited_structures:
        for s in visited_structures:
            if s != start_node:
                plt.scatter(s[0], s[1], c='orange', marker='X',
                            s=150, label='Visited Structure', zorder=5)

    # Axis settings
    plt.xlim(0, 16)
    plt.ylim(0, 16)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')

    # Legends
    area_legend_patches = []
    for area_val, color in area_colors.items():
        area_legend_patches.append(mpatches.Patch(
            color=color, label=f'Area {area_val}'))

    structure_legend_elements = [
        plt.Line2D(
            [0], [0],
            marker='o',
            color='w',
            label='Apartment/Building',
            markerfacecolor='brown',
            markersize=10
        ),
        plt.Line2D(
            [0], [0],
            marker='s',
            color='w',
            label='BandalgomCoffee',
            markerfacecolor='green',
            markersize=10
        ),
        plt.Line2D(
            [0], [0],
            marker='^',
            color='w',
            label='MyHome',
            markerfacecolor='green',
            markersize=10
        ),
        plt.Line2D(
            [0], [0],
            marker='s',
            color='w',
            label='Construction Site',
            markerfacecolor='grey',
            markersize=10
        ),
        plt.Line2D(
            [0], [0],
            color='red',
            lw=2,
            label='Path'
        )
    ]
    if extra_legend_elements:
        structure_legend_elements.extend(extra_legend_elements)

    legend1 = plt.legend(
        handles=area_legend_patches,
        loc='lower left',
        bbox_to_anchor=(0, 1.02, 0.5, 0.2),
        mode="expand",
        borderaxespad=0,
        title='Area Colors'
    )
    plt.gca().add_artist(legend1)

    legend2 = plt.legend(handles=structure_legend_elements, loc='lower right',
                         bbox_to_anchor=(0.5, 1.02, 0.5, 0.2), mode="expand",
                         borderaxespad=0,
                         title='Map Elements')

    plt.savefig(output_image_path)
    print(f'{output_image_path} 파일이 저장되었습니다.')


# --- Main Stage 3: MyHome to BandalgomCoffee ---
print("\n--- Stage 3: MyHome to BandalgomCoffee ---")
start_node_cafe = (
    df_merged[df_merged['struct'] == 'MyHome']['x'].iloc[0],
    df_merged[df_merged['struct'] == 'MyHome']['y'].iloc[0])
end_node_cafe = (
    df_merged[df_merged['struct'] == 'BandalgomCoffee']['x'].iloc[0],
    df_merged[df_merged['struct'] == 'BandalgomCoffee']['y'].iloc[0])
obstacles_cafe = set(zip(df_merged[df_merged['ConstructionSite'] == 1]['x'],
                         df_merged[df_merged['ConstructionSite'] == 1]['y']))

shortest_path_cafe = bfs(start_node_cafe, end_node_cafe, obstacles_cafe)

if shortest_path_cafe:
    path_df_cafe = pd.DataFrame(shortest_path_cafe, columns=['x', 'y'])
    output_csv_path_cafe = 'outputFiles/home_to_cafe.csv'
    path_df_cafe.to_csv(output_csv_path_cafe, index=False)
    print(f'{output_csv_path_cafe} 파일이 저장되었습니다.')

    plot_map(
        df_merged,
        shortest_path_cafe,
        'Path from MyHome to BandalgomCoffee',
        'outputFiles/map_final.png'
    )
else:
    print('MyHome에서 BandalgomCoffee까지의 경로를 찾을 수 없습니다.')

# --- Bonus Stage 3: Optimized Path Visiting All Structures ---
print("\n--- Bonus Stage 3: Optimized Path Visiting All Structures ---")


# Identify all unique structure locations (category > 0)
structure_coords_df = df_merged[(df_merged['category'] > 0) &
                                 (df_merged['ConstructionSite'] == 0)][
                                     ['x', 'y', 'struct']].drop_duplicates(
                                         subset=['x', 'y'])
all_structures = [
    (row['x'], row['y'])
    for index, row in structure_coords_df.iterrows()
]

my_home_coord_all = (
    df_merged[df_merged['struct'] == 'MyHome']['x'].iloc[0],
    df_merged[df_merged['struct'] == 'MyHome']['y'].iloc[0]
)

if my_home_coord_all not in all_structures:
    all_structures.append(my_home_coord_all)

structures_to_visit = [s for s in all_structures if s != my_home_coord_all]

# Path cache for BFS segments
path_cache = {}


def get_path_segment(start, end, obstacles):
    if (start, end) in path_cache:
        return path_cache[(start, end)]
    if (end, start) in path_cache:
        return [p for p in reversed(path_cache[(end, start)])]

    path = bfs(start, end, obstacles)
    if path:
        path_cache[(start, end)] = path
        return path
    return None


current_location_all = my_home_coord_all
visited_structures_all = {my_home_coord_all}
optimized_path_coords_all = [my_home_coord_all]

# Sort structures_to_visit for consistent behavior (optional)
structures_to_visit.sort()

while len(visited_structures_all) < len(all_structures):

    next_structure_all = None
    min_path_length_all = float('inf')
    shortest_segment_all = None

    for structure in structures_to_visit:
        if structure not in visited_structures_all:
            path_segment_all = get_path_segment(
                current_location_all, structure, obstacles_cafe)
            if (
                path_segment_all and 
                (len(path_segment_all) - 1) < min_path_length_all
            ):
                min_path_length_all = len(path_segment_all) - 1
                next_structure_all = structure
                shortest_segment_all = path_segment_all

    if next_structure_all:
        optimized_path_coords_all.extend(shortest_segment_all[1:])
        current_location_all = next_structure_all
        visited_structures_all.add(next_structure_all)
    else:
        print(
            f"Warning: Could not find path from {current_location_all} "
            f"to any unvisited structure. Optimized path might be incomplete."
        )
        break

if optimized_path_coords_all:
    optimized_path_df_all = pd.DataFrame(
        optimized_path_coords_all, columns=['x', 'y'])
    output_csv_path_all = 'outputFiles/optimized_all_structures_path.csv'
    optimized_path_df_all.to_csv(output_csv_path_all, index=False)
    print(f'{output_csv_path_all} 파일이 저장되었습니다.')

    extra_legend_elements_all = [
        plt.Line2D(
            [0], [0],
            marker='*',
            color='w',
            label='Path Start (MyHome)',
            markerfacecolor='blue',
            markersize=15
        ),
        plt.Line2D(
            [0], [0],
            marker='X',
            color='w',
            label='Visited Structure',
            markerfacecolor='orange',
            markersize=10
        )
    ]
    plot_map(
        df_merged, optimized_path_coords_all,
        'Optimized Path Visiting All Structures',
        'outputFiles/optimized_all_structures_path.png',
        extra_legend_elements=extra_legend_elements_all,
        start_node=my_home_coord_all,
        visited_structures=visited_structures_all
    )
else:
    print('최적화된 경로를 찾을 수 없습니다.')
