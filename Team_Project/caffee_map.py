import pandas as pd

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

# area 기준으로 정렬
df_merged = df_merged.sort_values(by='area').reset_index(drop=True)

# area 1 데이터 필터링
df_area1 = df_merged[df_merged['area'] == 1].copy()

# struct가 NaN인 경우 'None'으로 대체
df_area1['struct'] = df_area1['struct'].fillna('None')

# 결과 출력
print('Area 1 데이터:')
print(df_area1[['x', 'y', 'ConstructionSite', 'category', 'area', 'struct']])

# (보너스) 구조물 종류별 요약 통계 리포트
print('\n구조물 종류별 요약 통계:')
print(df_area1['struct'].value_counts())
