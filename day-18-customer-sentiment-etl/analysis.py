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

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import json
import os
import time
import anthropic
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

OUTPUT_DIR = 'output'
DB_FILE    = 'sentiment_reviews.db'
CHART_FILE = 'sentiment_trend.png'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Anthropic client ───────────────────────────────────────────
client = anthropic.Anthropic()

print('=' * 60)
print('CUSTOMER REVIEW SENTIMENT ETL PIPELINE')
print('=' * 60)


# ══════════════════════════════════════════════════════════════
# EXTRACT — Generate realistic product reviews with dates
# ══════════════════════════════════════════════════════════════
print('\n[EXTRACT] Generating product reviews...')

POSITIVE_REVIEWS = [
    "Absolutely love this product! Arrived quickly and exactly as described. Will definitely order again.",
    "Exceeded my expectations. The quality is outstanding and delivery was faster than promised.",
    "Great value for money. Works perfectly and the packaging was excellent. Highly recommend.",
    "Five stars. Customer service was helpful and the product is exactly what I needed.",
    "Brilliant product. Easy to set up and works flawlessly. Very happy with my purchase.",
    "Fantastic quality. My order arrived the next day and everything was in perfect condition.",
    "Really impressed with this. Better than the reviews suggested. Will be buying more.",
    "Perfect purchase. Exactly as described and arrived ahead of schedule. Thank you!",
    "Love it! The quality is incredible for the price. Will definitely be recommending to friends.",
    "Superb product. Packaged securely and delivered fast. Exactly what I wanted.",
    "Outstanding! The build quality is top notch and customer support was very responsive.",
    "Delighted with my purchase. Fast shipping, great product, no issues whatsoever.",
    "Excellent seller. Item was as described and dispatched quickly. Very satisfied.",
    "Could not be happier with this purchase. The product works great and looks even better.",
    "Amazing value. Premium quality at a fair price. Already recommended to three friends.",
]

NEUTRAL_REVIEWS = [
    "Product is fine. Does what it says on the box. Delivery was on time. Nothing special.",
    "It's okay. Works as expected but nothing remarkable. Packaging was a bit basic.",
    "Average product. Got what I paid for. Delivery was standard. No complaints.",
    "Decent enough. Does the job but I expected slightly better quality for the price.",
    "It arrived. Works. Not amazing but not bad either. Probably wouldn't reorder.",
    "Standard product. Delivery was fine. Instructions could be clearer but manageable.",
    "Does what it says. Nothing more. Delivery took the expected time. Fair enough.",
    "OK product. Took a while to arrive but it got here eventually. Nothing to write home about.",
    "Reasonable quality. Matches the description. Delivery was within the stated window.",
    "Not bad. Works as described. Packaging was minimal but the item arrived intact.",
]

NEGATIVE_REVIEWS = [
    "Terrible quality. Broke within a week of use. Complete waste of money. Avoid.",
    "Very disappointed. Item arrived damaged and customer service was unhelpful and slow.",
    "Do not buy this. Completely different from the description. Returning immediately.",
    "Awful experience. Delivery took three weeks and the product stopped working on day one.",
    "Poor quality. Looks nothing like the photos. Packaging was inadequate and it arrived scratched.",
    "Wasted my money. The product is cheap and flimsy. Customer service ignored my complaint.",
    "Shocking. Order arrived two weeks late with no communication. Product also faulty.",
    "Not as described at all. Size is wrong, colour is wrong, quality is terrible. Refund please.",
    "Worst purchase I've made online. Item missing parts and support takes days to respond.",
    "Extremely disappointed. Product failed after one use. Would give zero stars if I could.",
    "Broken on arrival. Replacement took another two weeks. Still not working properly.",
    "Misleading product description. What arrived looks nothing like what was advertised.",
    "Dreadful. Three weeks waiting, wrong item sent, and no apology from support.",
    "Defective product. The packaging was damaged and the item inside was broken.",
    "Terrible customer service. Contacted them four times and never got a resolution.",
]

# Generate 80 reviews spread across 6 months with realistic sentiment distribution
dates = pd.date_range('2024-01-01', '2024-06-30', periods=80)
rows = []

for i, date in enumerate(dates):
    # Simulate a trend: sentiment gets slightly worse mid-year then recovers
    month = date.month
    if month <= 2:
        weights = [0.60, 0.25, 0.15]
    elif month <= 4:
        weights = [0.45, 0.25, 0.30]
    else:
        weights = [0.55, 0.25, 0.20]

    sentiment_bucket = np.random.choice(['positive', 'neutral', 'negative'], p=weights)

    if sentiment_bucket == 'positive':
        text = POSITIVE_REVIEWS[np.random.randint(0, len(POSITIVE_REVIEWS))]
    elif sentiment_bucket == 'neutral':
        text = NEUTRAL_REVIEWS[np.random.randint(0, len(NEUTRAL_REVIEWS))]
    else:
        text = NEGATIVE_REVIEWS[np.random.randint(0, len(NEGATIVE_REVIEWS))]

    rows.append({
        'review_id'   : f'REV-{i+1:04d}',
        'date'        : date.strftime('%Y-%m-%d'),
        'review_text' : text,
        'star_rating' : {'positive': np.random.randint(4,6),
                         'neutral' : np.random.randint(3,4),
                         'negative': np.random.randint(1,3)}[sentiment_bucket]
    })

df_raw = pd.DataFrame(rows)
df_raw.to_csv('reviews_raw.csv', index=False)
print(f'  Reviews generated:    {len(df_raw)}')
print(f'  Date range:           {df_raw["date"].min()} to {df_raw["date"].max()}')
print(f'  Avg star rating:      {df_raw["star_rating"].mean():.2f}')


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
    """Send a batch of reviews to Claude and return parsed tags."""
    batch_input = json.dumps([
        {'review_id': r['review_id'], 'text': r['review_text']}
        for _, r in reviews_batch.iterrows()
    ])

    message = client.messages.create(
        model='claude-sonnet-4-20250514',
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{'role': 'user', 'content': batch_input}]
    )

    response_text = message.content[0].text.strip()
    # Strip markdown code fences if present
    if response_text.startswith('```'):
        response_text = response_text.split('```')[1]
        if response_text.startswith('json'):
            response_text = response_text[4:]
    return json.loads(response_text.strip())

# Process in batches of 10
BATCH_SIZE = 10
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
print(f'  3. Focus improvement on: {neg_themes.index[0].replace("_", " ")}')
print(f'  4. Run this pipeline weekly — early warning before trends worsen')
print('=' * 60)
