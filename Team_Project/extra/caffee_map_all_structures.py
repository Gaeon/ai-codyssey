import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import deque

# --- Data Loading ---
data_path = '../dataFile/'
map_file = f'{data_path}area_map.csv'
struct_file = f'{data_path}area_struct.csv'
category_file = f'{data_path}area_category.csv'

df_map = pd.read_csv(map_file, encoding='utf-8-sig')
df_struct = pd.read_csv(struct_file, encoding='utf-8-sig')
df_category = pd.read_csv(category_file, encoding='utf-8-sig')

df_category.columns = df_category.columns.str.strip()
df_category['struct'] = df_category['struct'].str.strip()

df_struct_with_category = pd.merge(
    df_struct, df_category, on='category', how='left')
df_merged = pd.merge(df_map, df_struct_with_category,
                     on=['x', 'y'], how='left')
df_merged['struct'] = df_merged['struct'].fillna('None')

# --- Obstacles and Structures Identification ---
obstacles = set(zip(df_merged[df_merged['ConstructionSite'] == 1]
                ['x'], df_merged[df_merged['ConstructionSite'] == 1]['y']))

# Identify all unique structure locations (category > 0)
# Exclude category 0 (empty space)
structure_coords_df = (
    df_merged[df_merged['category'] > 0][['x', 'y', 'struct']]
    .drop_duplicates(subset=['x', 'y'])
)
all_structures = [(row['x'], row['y'])
                  for index, row in structure_coords_df.iterrows()]

# Ensure MyHome is the starting point
my_home_coord = (
    df_merged[df_merged['struct'] == 'MyHome']['x'].iloc[0],
    df_merged[df_merged['struct'] == 'MyHome']['y'].iloc[0]
)

# Add MyHome to all_structures if it's not already there 
# (e.g., if its category was 0 initially)
if my_home_coord not in all_structures:
    all_structures.append(my_home_coord)

# Remove MyHome from the list of structures to visit initially, it's the start
structures_to_visit = [s for s in all_structures if s != my_home_coord]

# --- BFS for shortest path between two points ---


def bfs(start, end, obstacles, grid_size=15):
    queue = deque([[start]])
    visited = {start}

    # If start or end is an obstacle, no path
    if start in obstacles or end in obstacles:
        return None

    while queue:
        path = queue.popleft()
        x, y = path[-1]

        if (x, y) == end:
            return path

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:  # 4-directional movement
            nx, ny = x + dx, y + dy

            if (
                1 <= nx <= grid_size and 
                1 <= ny <= grid_size and 
                (nx, ny) not in visited and 
                (nx, ny) not in obstacles
            ):
                visited.add((nx, ny))
                new_path = list(path)
                new_path.append((nx, ny))
                queue.append(new_path)
    return None


# --- Calculate All-Pairs Shortest Paths ---
# This will store paths and their lengths to avoid re-calculating
path_cache = {}  # {(node1, node2): path_list}


def get_path_and_length(start, end):
    if (start, end) in path_cache:
        return path_cache[(start, end)]
    if (end, start) in path_cache:  # Path is bidirectional, reverse if needed
        return [p for p in reversed(path_cache[(end, start)])]

    path = bfs(start, end, obstacles)
    if path:
        path_cache[(start, end)] = path
        return path
    return None


# --- Nearest Neighbor TSP Heuristic ---
current_location = my_home_coord
visited_structures = {my_home_coord}
optimized_path_coords = [my_home_coord]

# Sort structures_to_visit for consistent behavior (optional)
structures_to_visit.sort()

while len(visited_structures) < len(all_structures):
    next_structure = None
    min_path_length = float('inf')
    shortest_segment = None

    for structure in structures_to_visit:
        if structure not in visited_structures:
            path_segment = get_path_and_length(current_location, structure)
            # -1 because path includes start and end
            if path_segment and (len(path_segment) - 1) < min_path_length:
                min_path_length = len(path_segment) - 1
                next_structure = structure
                shortest_segment = path_segment

    if next_structure:
        # Add the path segment 
        # (excluding the start node of the segment, as it's current_location)
        optimized_path_coords.extend(shortest_segment[1:])
        current_location = next_structure
        visited_structures.add(next_structure)
    else:
        print(
            f"Warning: Could not find path from {current_location} "
            f"to any unvisited structure. Path might be incomplete."
        )
        break

# --- Visualization ---
plt.figure(figsize=(10, 12))  # Increased figure height for legends
plt.grid(True)

# Plot all cells with area colors (background)
area_colors = {
    0: 'lightgrey',
    1: 'lightblue',
    2: 'lightgreen',
    3: 'lightyellow'
}
for index, row in df_merged.iterrows():
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
for idx, row in df_merged.iterrows():
    if row['struct'] == 'Apartment' or row['struct'] == 'Building':
        plt.scatter(
            row['x'], row['y'],
            c='brown',
            marker='o',
            zorder=3
        )
    elif row['struct'] == 'BandalgomCoffee':
        plt.scatter(
            row['x'], row['y'],
            c='green',
            marker='s',
            zorder=3
        )
    elif row['struct'] == 'MyHome':
        plt.scatter(
            row['x'], row['y'],
            c='green',
            marker='^',
            zorder=3
        )

construction_sites = df_merged[df_merged['ConstructionSite'] == 1]
for idx, row in construction_sites.iterrows():
    plt.scatter(
        row['x'], row['y'],
        c='grey',
        marker='s',
        s=200,
        zorder=4
    )  # Construction sites on top

# Plot the optimized path
if optimized_path_coords:
    path_x, path_y = zip(*optimized_path_coords)
    # Path below structures but above background
    plt.plot(
        path_x, path_y,
        c='red',
        linewidth=2,
        zorder=2
    )

# Plot start and end points of the overall path
plt.scatter(
    my_home_coord[0], my_home_coord[1],
    c='blue',
    marker='*',
    s=300,
    label='Start (MyHome)',
    zorder=5
)
# Mark other structures visited
for s in all_structures:
    if s != my_home_coord:
        plt.scatter(
            s[0], s[1],
            c='orange',
            marker='X',
            s=150,
            label='Visited Structure',
            zorder=5
        )


# Axis settings
plt.xlim(0, 16)
plt.ylim(0, 16)
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal', adjustable='box')

# Legends
area_legend_patches = []
for area_val, color in area_colors.items():
    area_legend_patches.append(
        mpatches.Patch(
            color=color,
            label=f'Area {area_val}'
        )
    )

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
        label='Optimized Path'
    ),
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

# Combine legends or place them strategically
legend1 = plt.legend(
    handles=area_legend_patches,
    loc='lower left',
    bbox_to_anchor=(0, 1.02, 0.5, 0.2),
    mode="expand",
    borderaxespad=0,
    title='Area Colors'
)
# Add the first legend manually so the second one doesn't overwrite it
plt.gca().add_artist(legend1)

legend2 = plt.legend(
    handles=structure_legend_elements,
    loc='lower right',
    bbox_to_anchor=(0.5, 1.02, 0.5, 0.2),
    mode="expand",
    borderaxespad=0,
    title='Map Elements'
)

# Save the image
output_image_path = 'optimized_all_structures_path.png'
plt.savefig(output_image_path)
print(f'{output_image_path} 파일이 저장되었습니다.')

# Save the optimized path to CSV
if optimized_path_coords:
    optimized_path_df = pd.DataFrame(optimized_path_coords, columns=['x', 'y'])
    output_csv_path = 'optimized_all_structures_path.csv'
    optimized_path_df.to_csv(output_csv_path, index=False)
    print(f'{output_csv_path} 파일이 저장되었습니다.')
else:
    print('최적화된 경로를 찾을 수 없습니다.')