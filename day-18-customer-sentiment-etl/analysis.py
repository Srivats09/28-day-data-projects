"""
etl_pipeline.py
===============
Day 18: Customer Review Sentiment ETL Pipeline

Industry:  Marketing / Retail
Format:    Python script (.py)
Skills:    ETL, pandas, Anthropic API, sqlite3, matplotlib, JSON parsing

Who uses this:
    A CX manager presenting the weekly voice-of-customer report
    to leadership. Instead of reading 200 reviews manually, this
    pipeline tags every review with sentiment, theme, and urgency
    using Claude — then visualises the trend over time.

Data:
    Synthetic product reviews — realistic e-commerce review text
    with dates spread across 6 months. Same pipeline runs on any
    real review export (Trustpilot, Google, Amazon) by swapping
    the CSV.
"""

from email import message

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import json
import os
import time
from groq import Groq
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

OUTPUT_DIR = 'output'
DB_FILE    = 'sentiment_reviews.db'
CHART_FILE = 'sentiment_trend.png'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Anthropic client ───────────────────────────────────────────
api_key = os.environ.get('GROQ_API_KEY')
if not api_key:
    raise ValueError('GROQ_API_KEY not set. Run: $env:GROQ_API_KEY = "your-key"')
client = Groq(api_key=api_key)

print('=' * 60)
print('CUSTOMER REVIEW SENTIMENT ETL PIPELINE')
print('=' * 60)


# ══════════════════════════════════════════════════════════════
# EXTRACT — Generate realistic product reviews with dates
# ══════════════════════════════════════════════════════════════
print('\n[EXTRACT] Loading TripAdvisor hotel reviews...')

df_raw = pd.read_csv('reviews.csv')
df_raw.columns = ['review_text', 'star_rating']

# Sample 200 reviews for API rate limits
df_raw = df_raw.dropna(subset=['review_text'])
df_raw = df_raw.sample(200, random_state=42).reset_index(drop=True)

# Generate review IDs and synthetic dates spread over 6 months
df_raw['review_id'] = [f'REV-{i+1:04d}' for i in range(len(df_raw))]
dates = pd.date_range('2024-01-01', '2024-06-30', periods=200)
df_raw['date'] = [d.strftime('%Y-%m-%d') for d in dates]

# Convert 0/1 to readable rating
df_raw['star_rating'] = df_raw['star_rating'].map({1: 5, 0: 1})

# Trim long reviews to avoid token limits
df_raw['review_text'] = df_raw['review_text'].str[:500]

df_raw.to_csv('reviews_sampled.csv', index=False)
print(f'  Reviews loaded:       {len(df_raw):,} (sampled from full dataset)')
print(f'  Date range:           {df_raw["date"].min()} to {df_raw["date"].max()}')
print(f'  Rating distribution:')
print(df_raw['star_rating'].value_counts().to_string())


# ══════════════════════════════════════════════════════════════
# TRANSFORM — Batch-send to Claude for sentiment + theme tagging
# ══════════════════════════════════════════════════════════════
print('\n[TRANSFORM] Tagging reviews with Claude API...')
print('  Sending in batches of 10...')
start = time.time()

SYSTEM_PROMPT = """You are a CX analyst. For each review provided, return ONLY a JSON array.
Each element must have exactly these fields:
- review_id: string (copy from input)
- sentiment: string — exactly one of: "positive", "neutral", "negative"
- theme: string — exactly one of: "product_quality", "delivery", "customer_service", "value_for_money", "packaging", "general"
- urgency: string — exactly one of: "high", "medium", "low"
- key_phrase: string — 3-6 word summary of the main point

Return ONLY the JSON array. No other text."""

def tag_batch(reviews_batch):
    batch_input = json.dumps([
        {'review_id': r['review_id'], 'text': r['review_text']}
        for _, r in reviews_batch.iterrows()
    ])

    try:
        message = client.chat.completions.create(
            model='llama-3.1-8b-instant',
            max_tokens=1500,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': batch_input}
            ]
        )
        response_text = message.choices[0].message.content.strip()
        if response_text.startswith('```'):
            response_text = response_text.split('```')[1]
            if response_text.startswith('json'):
                response_text = response_text[4:]
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        # Fallback — tag individually
        results = []
        for _, row in reviews_batch.iterrows():
            try:
                single = client.chat.completions.create(
                    model='llama-3.1-8b-instant',
                    max_tokens=200,
                    messages=[
                        {'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': json.dumps([
                            {'review_id': row['review_id'], 'text': row['review_text']}
                        ])}
                    ]
                )
                text = single.choices[0].message.content.strip()
                if text.startswith('```'):
                    text = text.split('```')[1]
                    if text.startswith('json'):
                        text = text[4:]
                parsed = json.loads(text.strip())
                results.extend(parsed if isinstance(parsed, list) else [parsed])
            except Exception:
                results.append({
                    'review_id': row['review_id'],
                    'sentiment': 'neutral',
                    'theme': 'general',
                    'urgency': 'low',
                    'key_phrase': 'parse error'
                })
        return results

# Process in batches of 10
BATCH_SIZE = 5
all_tags = []
total_batches = (len(df_raw) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_num in range(total_batches):
    batch_start = batch_num * BATCH_SIZE
    batch_end   = min(batch_start + BATCH_SIZE, len(df_raw))
    batch       = df_raw.iloc[batch_start:batch_end]

    print(f'  Batch {batch_num+1}/{total_batches} '
          f'(reviews {batch_start+1}-{batch_end})...', end=' ', flush=True)

    try:
        tags = tag_batch(batch)
        all_tags.extend(tags)
        print(f'✓ {len(tags)} tagged')
    except Exception as e:
        print(f'✗ Error: {e}')
        # Fallback: mark as untagged
        for _, row in batch.iterrows():
            all_tags.append({
                'review_id': row['review_id'],
                'sentiment': 'neutral',
                'theme': 'general',
                'urgency': 'low',
                'key_phrase': 'tagging failed'
            })

    time.sleep(0.5)  # rate limit safety

df_tags = pd.DataFrame(all_tags)
df = df_raw.merge(df_tags, on='review_id', how='left')
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.to_period('M').astype(str)
df['week']  = df['date'].dt.to_period('W').apply(lambda r: r.start_time).dt.strftime('%Y-%m-%d')

print(f'\n  Total tagged:         {len(df_tags)}')
print(f'  Time taken:           {round(time.time()-start, 1)}s')
print(f'\n  Sentiment breakdown:')
print(df['sentiment'].value_counts().to_string())
print(f'\n  Theme breakdown:')
print(df['theme'].value_counts().to_string())
print(f'\n  Urgency breakdown:')
print(df['urgency'].value_counts().to_string())


# ══════════════════════════════════════════════════════════════
# LOAD — Insert tagged reviews into SQLite
# ══════════════════════════════════════════════════════════════
print('\n[LOAD] Writing to SQLite...')

conn = sqlite3.connect(DB_FILE)
df.to_sql('reviews', conn, if_exists='replace', index=False)

# Monthly sentiment summary view
monthly_summary = (
    df.groupby(['month', 'sentiment'])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)
for col in ['positive', 'neutral', 'negative']:
    if col not in monthly_summary.columns:
        monthly_summary[col] = 0
monthly_summary['total']     = monthly_summary[['positive','neutral','negative']].sum(axis=1)
monthly_summary['pos_rate']  = (monthly_summary['positive'] / monthly_summary['total'] * 100).round(1)
monthly_summary['neg_rate']  = (monthly_summary['negative'] / monthly_summary['total'] * 100).round(1)
monthly_summary.to_sql('monthly_sentiment', conn, if_exists='replace', index=False)

print(f'  reviews table:        {len(df)} rows')
print(f'  monthly_sentiment:    {len(monthly_summary)} rows')

# High urgency negatives — the action list
action_list = df[
    (df['sentiment'] == 'negative') & (df['urgency'] == 'high')
][['review_id','date','theme','key_phrase','star_rating','review_text']].sort_values('date')
action_list.to_sql('action_list', conn, if_exists='replace', index=False)
print(f'  action_list:          {len(action_list)} high-urgency negatives')


# ══════════════════════════════════════════════════════════════
# VISUALISE
# ══════════════════════════════════════════════════════════════
print('\n[VISUALISE] Building dashboard...')

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Customer Review Sentiment Analysis — Jan to Jun 2024',
             fontsize=14, fontweight='bold', y=1.01)

sentiment_colors = {
    'positive': '#1D9E75',
    'neutral' : '#EF9F27',
    'negative': '#E24B4A'
}

# Panel 1 — Monthly sentiment trend (stacked bar)
months = monthly_summary['month'].tolist()
x = range(len(months))
axes[0,0].bar(x, monthly_summary['positive'], label='Positive', color='#1D9E75')
axes[0,0].bar(x, monthly_summary['neutral'],  label='Neutral',  color='#EF9F27',
              bottom=monthly_summary['positive'])
axes[0,0].bar(x, monthly_summary['negative'], label='Negative', color='#E24B4A',
              bottom=monthly_summary['positive'] + monthly_summary['neutral'])
axes[0,0].set_xticks(x)
axes[0,0].set_xticklabels(months, rotation=20, fontsize=9)
axes[0,0].set_ylabel('Number of reviews')
axes[0,0].set_title('Monthly review volume by sentiment')
axes[0,0].legend(fontsize=9)

# Panel 2 — Positive rate trend line
axes[0,1].plot(x, monthly_summary['pos_rate'], color='#1D9E75',
               linewidth=2.5, marker='o', markersize=7, label='Positive rate %')
axes[0,1].plot(x, monthly_summary['neg_rate'], color='#E24B4A',
               linewidth=2.5, marker='s', markersize=7, label='Negative rate %')
axes[0,1].set_xticks(x)
axes[0,1].set_xticklabels(months, rotation=20, fontsize=9)
axes[0,1].set_ylabel('% of reviews')
axes[0,1].set_title('Sentiment rate trend over time')
axes[0,1].legend(fontsize=9)
axes[0,1].axhline(50, color='gray', linestyle='--', linewidth=0.8)

# Panel 3 — Theme distribution
theme_counts = df['theme'].value_counts()
theme_colors = ['#378ADD','#1D9E75','#EF9F27','#E24B4A','#534AB7','#888780']
axes[1,0].barh(theme_counts.index, theme_counts.values,
               color=theme_colors[:len(theme_counts)])
axes[1,0].set_xlabel('Number of reviews')
axes[1,0].set_title('Reviews by theme')
axes[1,0].invert_yaxis()

# Panel 4 — Sentiment by theme heatmap
theme_sent = df.groupby(['theme','sentiment']).size().unstack(fill_value=0)
for col in ['positive','neutral','negative']:
    if col not in theme_sent.columns:
        theme_sent[col] = 0
theme_sent = theme_sent[['positive','neutral','negative']]
import seaborn as sns
sns.heatmap(theme_sent, ax=axes[1,1], cmap='RdYlGn', annot=True,
            fmt='d', linewidths=0.5, cbar_kws={'label': 'Review count'})
axes[1,1].set_title('Sentiment by theme heatmap')
axes[1,1].set_xlabel('')

plt.tight_layout()
plt.savefig(CHART_FILE, dpi=150, bbox_inches='tight')
print(f'  Chart saved as {CHART_FILE}')
plt.show()


# ══════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════
df.to_csv(f'{OUTPUT_DIR}/tagged_reviews.csv', index=False)
action_list.to_csv(f'{OUTPUT_DIR}/action_list.csv', index=False)
monthly_summary.to_csv(f'{OUTPUT_DIR}/monthly_summary.csv', index=False)

conn.close()

total          = len(df)
n_pos          = (df['sentiment'] == 'positive').sum()
n_neg          = (df['sentiment'] == 'negative').sum()
n_neu          = (df['sentiment'] == 'neutral').sum()
n_high_urgency = (df['urgency'] == 'high').sum()
top_theme      = df['theme'].value_counts().index[0]
worst_month    = monthly_summary.loc[monthly_summary['neg_rate'].idxmax(), 'month']
best_month     = monthly_summary.loc[monthly_summary['pos_rate'].idxmax(), 'month']

print('\n' + '=' * 60)
print('BUSINESS INSIGHT SUMMARY')
print('=' * 60)
print(f'Total reviews tagged:      {total}')
print(f'Positive:                  {n_pos} ({n_pos/total*100:.1f}%)')
print(f'Neutral:                   {n_neu} ({n_neu/total*100:.1f}%)')
print(f'Negative:                  {n_neg} ({n_neg/total*100:.1f}%)')
print(f'High urgency reviews:      {n_high_urgency}')
print()
print(f'Most common theme:         {top_theme}')
print(f'Best month (pos rate):     {best_month} ({monthly_summary.loc[monthly_summary["pos_rate"].idxmax(), "pos_rate"]}%)')
print(f'Worst month (neg rate):    {worst_month} ({monthly_summary.loc[monthly_summary["neg_rate"].idxmax(), "neg_rate"]}%)')
print()
print('TOP COMPLAINT THEMES (negative reviews):')
neg_themes = df[df['sentiment']=='negative']['theme'].value_counts().head(3)
for theme, count in neg_themes.items():
    print(f'  {theme:25s}: {count} reviews')
print()
print('CX RECOMMENDATIONS:')
print(f'  1. Address {n_high_urgency} high-urgency negative reviews immediately')
print(f'  2. Investigate {worst_month} — highest negative rate of the period')
if len(neg_themes) > 0:
    print(f'  3. Focus improvement on: {neg_themes.index[0].replace("_", " ")}')
print(f'  4. Run this pipeline weekly — early warning before trends worsen')
print('=' * 60)
