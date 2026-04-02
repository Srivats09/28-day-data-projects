"""
elt_pipeline.py
===============
Day 10: Streaming Platform Content ELT Pipeline

Industry:  Media / Entertainment
Format:    Python script (.py)
Skills:    ELT, pandas, sqlite3, SQL views, matplotlib

ELT vs ETL:
    ETL — transform data BEFORE loading into the database
    ELT — load raw data FIRST, then transform using SQL views
    This project uses the ELT pattern — raw data goes into SQLite
    unchanged, then SQL views handle all transformations in-place.
    This is how modern data warehouses (BigQuery, Snowflake) work.

Who uses this:
    A content acquisition manager deciding which genres and countries
    to commission next. This pipeline answers: what performs best,
    what has grown fastest, and where the content gaps are.
"""

import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import os
import time
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = 'output'
DB_FILE    = 'streaming_content.db'
CHART_FILE = 'content_analysis.png'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print('=' * 60)
print('STREAMING CONTENT ELT PIPELINE')
print('=' * 60)


# ══════════════════════════════════════════════════════════════
# EXTRACT + LOAD (ELT — load raw, transform in SQL)
# ══════════════════════════════════════════════════════════════
print('\n[EXTRACT] Loading raw Netflix titles...')
start = time.time()

df_raw = pd.read_csv('netflix_titles.csv')
print(f'  Raw rows: {len(df_raw):,} | Columns: {len(df_raw.columns)}')
print(f'  Nulls per column:')
print(df_raw.isnull().sum().to_string())

print(f'\n[LOAD] Writing raw data to SQLite (ELT — no transforms yet)...')
conn = sqlite3.connect(DB_FILE)
df_raw.to_sql('raw_titles', conn, if_exists='replace', index=False)
print(f'  raw_titles table: {len(df_raw):,} rows loaded')
print(f'  Time taken: {round(time.time()-start, 2)}s')


# ══════════════════════════════════════════════════════════════
# TRANSFORM — create SQL views (ELT pattern)
# Views transform data in-place without duplicating storage
# ══════════════════════════════════════════════════════════════
print('\n[TRANSFORM] Creating SQL views...')

conn.execute('DROP VIEW IF EXISTS v_content_clean')
conn.execute('''
CREATE VIEW v_content_clean AS
SELECT
    show_id,
    type,
    title,
    TRIM(director)                          AS director,
    country,
    TRIM(date_added)                        AS date_added,
    CAST(release_year AS INTEGER)           AS release_year,
    rating,
    duration,
    listed_in                               AS genres,
    description,
    CASE
        WHEN type = 'Movie' THEN
            CAST(REPLACE(duration, ' min', '') AS INTEGER)
        ELSE NULL
    END                                     AS duration_mins,
    CASE
        WHEN type = 'TV Show' THEN
            CAST(REPLACE(duration, ' Season', '')
                     AS INTEGER)
        ELSE NULL
    END                                     AS seasons,
    SUBSTR(TRIM(date_added), -4, 4)         AS year_added
FROM raw_titles
WHERE title IS NOT NULL
  AND type  IS NOT NULL
''')

conn.execute('DROP VIEW IF EXISTS v_genre_performance')
conn.execute('''
CREATE VIEW v_genre_performance AS
SELECT
    TRIM(genre_raw.value)  AS genre,
    c.type                 AS content_type,
    COUNT(*)               AS title_count,
    AVG(CASE WHEN c.type = 'Movie'
             THEN CAST(REPLACE(c.duration, ' min', '') AS INTEGER)
             ELSE NULL END) AS avg_movie_duration_mins
FROM v_content_clean c,
     json_each('["' || REPLACE(c.genres, ', ', '","') || '"]') AS genre_raw
GROUP BY genre, c.type
HAVING title_count >= 5
''')

conn.execute('DROP VIEW IF EXISTS v_country_performance')
conn.execute('''             
CREATE VIEW v_country_performance AS
SELECT
    TRIM(c.value)  AS country,
    COUNT(DISTINCT v.show_id) AS title_count,
    SUM(CASE WHEN v.type = 'Movie'   THEN 1 ELSE 0 END) AS movies,
    SUM(CASE WHEN v.type = 'TV Show' THEN 1 ELSE 0 END) AS tv_shows
FROM v_content_clean v,
     json_each('["' || REPLACE(v.country, ', ', '","') || '"]') AS c
WHERE v.country IS NOT NULL
  AND TRIM(c.value) != ''
GROUP BY TRIM(c.value)
HAVING title_count >= 10
ORDER BY title_count DESC
''')

conn.commit()
print('  v_content_clean      — cleaned titles view')
print('  v_genre_performance  — genre stats view')
print('  v_country_performance — country stats view')


# ══════════════════════════════════════════════════════════════
# ANALYSE — query the views
# ══════════════════════════════════════════════════════════════
print('\n[ANALYSE] Running analysis queries on views...')

# Q1 — Content type split
q1 = pd.read_sql_query('''
    SELECT type, COUNT(*) AS count,
           ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM v_content_clean), 1) AS pct
    FROM v_content_clean
    GROUP BY type
''', conn)

# Q2 — Content added per year trend
q2 = pd.read_sql_query('''
    SELECT year_added, type, COUNT(*) AS titles
    FROM v_content_clean
    WHERE year_added IS NOT NULL
      AND year_added != ''
      AND CAST(year_added AS INTEGER) BETWEEN 2015 AND 2021
    GROUP BY year_added, type
    ORDER BY year_added, type
''', conn)

# Q3 — Top 15 genres by title count
q3 = pd.read_sql_query('''
    SELECT genre, SUM(title_count) AS total_titles
    FROM v_genre_performance
    GROUP BY genre
    ORDER BY total_titles DESC
    LIMIT 15
''', conn)

# Q4 — Top 15 countries by content volume
q4 = pd.read_sql_query('''
    SELECT country, title_count, movies, tv_shows
    FROM v_country_performance
    LIMIT 15
''', conn)

# Q5 — Movie duration distribution
q5 = pd.read_sql_query('''
    SELECT
        CASE
            WHEN duration_mins < 60  THEN 'Under 60 min'
            WHEN duration_mins < 90  THEN '60-90 min'
            WHEN duration_mins < 120 THEN '90-120 min'
            WHEN duration_mins < 150 THEN '120-150 min'
            ELSE 'Over 150 min'
        END AS duration_band,
        COUNT(*) AS movies
    FROM v_content_clean
    WHERE type = 'Movie' AND duration_mins IS NOT NULL
    GROUP BY duration_band
    ORDER BY MIN(duration_mins)
''', conn)

# Q6 — Rating distribution
q6 = pd.read_sql_query('''
    SELECT rating, COUNT(*) AS count
    FROM v_content_clean
    WHERE rating IS NOT NULL
      AND rating NOT IN ('74 min', '84 min', '66 min')
    GROUP BY rating
    ORDER BY count DESC
    LIMIT 10
''', conn)

print('  Queries complete')
print('\n=== Content type split ===')
print(q1.to_string())
print('\n=== Top 15 genres ===')
print(q3.to_string())
print('\n=== Top 10 countries ===')
print(q4.head(10).to_string())


# ══════════════════════════════════════════════════════════════
# VISUALISE
# ══════════════════════════════════════════════════════════════
print('\n[VISUALISE] Building dashboard...')

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle('Netflix Content Library Analysis — ELT Pipeline',
             fontsize=14, fontweight='bold', y=1.01)

# Panel 1 — Top 15 genres
axes[0, 0].barh(q3['genre'], q3['total_titles'], color='#E24B4A')
axes[0, 0].set_xlabel('Number of titles')
axes[0, 0].set_title('Top 15 genres by title count')
axes[0, 0].invert_yaxis()
axes[0, 0].tick_params(axis='y', labelsize=8)

# Panel 2 — Content added per year (stacked bar)
q2_pivot = q2.pivot(index='year_added', columns='type', values='titles').fillna(0)
q2_pivot.plot(kind='bar', ax=axes[0, 1], color=['#378ADD', '#E24B4A'],
              width=0.7, stacked=False)
axes[0, 1].set_xlabel('Year added')
axes[0, 1].set_ylabel('Titles added')
axes[0, 1].set_title('Content added per year by type')
axes[0, 1].tick_params(axis='x', rotation=20)
axes[0, 1].legend(fontsize=9)

# Panel 3 — Top 15 countries
axes[1, 0].barh(q4['country'], q4['title_count'], color='#1D9E75')
axes[1, 0].set_xlabel('Total titles')
axes[1, 0].set_title('Top 15 content-producing countries')
axes[1, 0].invert_yaxis()
axes[1, 0].tick_params(axis='y', labelsize=8)

# Panel 4 — Movie duration distribution
dur_order = ['Under 60 min', '60-90 min', '90-120 min', '120-150 min', 'Over 150 min']
q5_plot = q5.set_index('duration_band').reindex(dur_order).reset_index()
axes[1, 1].bar(q5_plot['duration_band'], q5_plot['movies'], color='#EF9F27')
axes[1, 1].set_ylabel('Number of movies')
axes[1, 1].set_title('Movie duration distribution')
axes[1, 1].tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.savefig(CHART_FILE, dpi=150, bbox_inches='tight')
print(f'  Chart saved as {CHART_FILE}')
plt.show()


# ══════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════
q3.to_csv(f'{OUTPUT_DIR}/genre_performance.csv', index=False)
q4.to_csv(f'{OUTPUT_DIR}/country_performance.csv', index=False)
q5.to_csv(f'{OUTPUT_DIR}/movie_duration_distribution.csv', index=False)
q6.to_csv(f'{OUTPUT_DIR}/rating_distribution.csv', index=False)

total     = pd.read_sql_query('SELECT COUNT(*) as c FROM v_content_clean', conn).iloc[0]['c']
movies    = q1[q1['type']=='Movie']['count'].values[0]
shows     = q1[q1['type']=='TV Show']['count'].values[0]
top_genre = q3.iloc[0]['genre']
top_ctry  = q4.iloc[0]['country']

conn.close()

print('\n' + '=' * 60)
print('BUSINESS INSIGHT SUMMARY')
print('=' * 60)
print(f'Total titles in library:   {total:,}')
print(f'Movies:                    {movies:,} ({q1[q1["type"]=="Movie"]["pct"].values[0]}%)')
print(f'TV Shows:                  {shows:,} ({q1[q1["type"]=="TV Show"]["pct"].values[0]}%)')
print(f'Countries producing content: {len(q4)}')
print()
print(f'Top genre:                 {top_genre} ({q3.iloc[0]["total_titles"]} titles)')
print(f'Top producing country:     {top_ctry} ({q4.iloc[0]["title_count"]} titles)')
print(f'Most common movie length:  90-120 mins ({q5_plot[q5_plot["duration_band"]=="90-120 min"]["movies"].values[0]:,} movies)')
print()
print('ELT PATTERN SUMMARY:')
print('  Raw data loaded to SQLite unchanged')
print('  3 SQL views handle all transformations in-place')
print('  No data duplication — views query raw table directly')
print('  Same pattern used in BigQuery, Snowflake, Redshift')
print()
print('CONTENT ACQUISITION RECOMMENDATIONS:')
print(f'  1. Double down on {top_genre} — largest genre by volume')
print(f'  2. Expand beyond {top_ctry} — diversify production base')
print(f'  3. 90-120 min sweet spot — most common movie length')
print(f'  4. TV Shows are {q1[q1["type"]=="TV Show"]["pct"].values[0]}% of library — consider increasing to drive retention')
print('=' * 60)
