"""
Day 04 — Smart Meter Energy ETL Pipeline
Dataset: UCI Household Power Consumption (~2M minute-level readings, 2006–2010)
Real-world problem: Utilities & households need clean, queryable energy data
to detect waste, flag anomalies, and reduce bills.
Pipeline: Extract → Clean → Transform → Load (SQLite) → Business Insights
"""

import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import os

# ── CONFIG ───────────────────────────────────────────────────────────────────
RAW_FILE  = "household_power_consumption.txt"
DB_FILE   = "smart_meter.db"
CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)

# ── STEP 1: EXTRACT ──────────────────────────────────────────────────────────
print("=" * 55)
print("STEP 1: Extracting raw data...")
df = pd.read_csv(RAW_FILE, sep=";", na_values=["?"], low_memory=False)
print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")

# ── STEP 2: TRANSFORM ────────────────────────────────────────────────────────
print("STEP 2: Transforming & enriching...")

df["datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
)
df.drop(columns=["Date", "Time"], inplace=True)

num_cols = [
    "Global_active_power", "Global_reactive_power", "Voltage",
    "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"
]
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

before = len(df)
df.dropna(subset=["datetime", "Global_active_power"], inplace=True)
print(f"  Dropped {before - len(df):,} rows with missing critical values")
print(f"  Clean rows: {len(df):,}")

# Derived columns
df["year"]        = df["datetime"].dt.year
df["month"]       = df["datetime"].dt.month
df["hour"]        = df["datetime"].dt.hour
df["day_of_week"] = df["datetime"].dt.day_name()
df["is_weekend"]  = df["datetime"].dt.weekday >= 5

# Energy in kWh per minute
df["energy_kwh"] = df["Global_active_power"] / 60

# Unmetered = total minus 3 sub-meters (converted to same units)
df["sub_metering_untracked"] = (
    df["Global_active_power"] * 1000 / 60
    - df["Sub_metering_1"]
    - df["Sub_metering_2"]
    - df["Sub_metering_3"]
).clip(lower=0)

# Anomaly flag: power > mean + 3 std deviations
mean_p = df["Global_active_power"].mean()
std_p  = df["Global_active_power"].std()
df["is_anomaly"] = df["Global_active_power"] > (mean_p + 3 * std_p)

print(f"  Anomalies flagged: {df['is_anomaly'].sum():,} readings")

# ── STEP 3: LOAD INTO SQLITE ─────────────────────────────────────────────────
print("STEP 3: Loading into SQLite...")
conn = sqlite3.connect(DB_FILE)

df.to_sql("readings", conn, if_exists="replace", index=False)

conn.executescript("""
    DROP VIEW IF EXISTS monthly_summary;
    CREATE VIEW monthly_summary AS
    SELECT
        year, month,
        ROUND(SUM(energy_kwh), 2)           AS total_kwh,
        ROUND(AVG(Global_active_power), 3)  AS avg_power_kw,
        ROUND(MAX(Global_active_power), 3)  AS peak_power_kw,
        ROUND(AVG(Voltage), 2)              AS avg_voltage,
        SUM(is_anomaly)                     AS anomaly_count,
        COUNT(*)                            AS readings
    FROM readings
    GROUP BY year, month
    ORDER BY year, month;

    DROP VIEW IF EXISTS hourly_profile;
    CREATE VIEW hourly_profile AS
    SELECT
        hour,
        ROUND(AVG(Global_active_power), 3) AS avg_power_kw,
        ROUND(AVG(CASE WHEN is_weekend=1 THEN Global_active_power END), 3) AS weekend_avg_kw,
        ROUND(AVG(CASE WHEN is_weekend=0 THEN Global_active_power END), 3) AS weekday_avg_kw
    FROM readings
    GROUP BY hour
    ORDER BY hour;

    DROP VIEW IF EXISTS submeter_breakdown;
    CREATE VIEW submeter_breakdown AS
    SELECT
        year,
        ROUND(SUM(Sub_metering_1)/1000.0/60, 1) AS kitchen_kwh,
        ROUND(SUM(Sub_metering_2)/1000.0/60, 1) AS laundry_kwh,
        ROUND(SUM(Sub_metering_3)/1000.0/60, 1) AS hvac_kwh,
        ROUND(SUM(sub_metering_untracked)/1000.0/60, 1) AS untracked_kwh
    FROM readings
    GROUP BY year
    ORDER BY year;
""")
conn.commit()
print(f"  Saved to {DB_FILE} | 3 SQL views created")

# ── STEP 4: VALIDATE ─────────────────────────────────────────────────────────
print("STEP 4: Querying insights from database...")
monthly   = pd.read_sql("SELECT * FROM monthly_summary", conn)
hourly    = pd.read_sql("SELECT * FROM hourly_profile", conn)
submeter  = pd.read_sql("SELECT * FROM submeter_breakdown", conn)

# ── STEP 5: VISUALISE ────────────────────────────────────────────────────────
print("STEP 5: Generating charts...")

# Chart 1 — Monthly energy consumption
fig, ax = plt.subplots(figsize=(13, 4))
labels = [f"{int(r.year)}-{int(r.month):02d}" for _, r in monthly.iterrows()]
bars = ax.bar(labels, monthly["total_kwh"], color="#2563eb")
anomaly_months = monthly[monthly["anomaly_count"] > 50]
for _, row in anomaly_months.iterrows():
    label = f"{int(row.year)}-{int(row.month):02d}"
    if label in labels:
        bars[labels.index(label)].set_color("#dc2626")
ax.set_title("Monthly Household Energy Consumption (kWh) — Red = High Anomaly Month", fontsize=12)
ax.set_xlabel("Month"); ax.set_ylabel("kWh")
plt.xticks(rotation=45, ha="right", fontsize=7)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/monthly_consumption.png", dpi=150)
plt.close()

# Chart 2 — Weekday vs Weekend hourly profile
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(hourly["hour"], hourly["weekday_avg_kw"], label="Weekday", color="#2563eb", linewidth=2, marker="o", markersize=4)
ax.plot(hourly["hour"], hourly["weekend_avg_kw"], label="Weekend", color="#16a34a", linewidth=2, marker="s", markersize=4)
ax.fill_between(hourly["hour"], hourly["weekday_avg_kw"], hourly["weekend_avg_kw"], alpha=0.1, color="gray")
ax.set_title("Average Power Demand: Weekday vs Weekend by Hour (kW)", fontsize=12)
ax.set_xlabel("Hour of Day"); ax.set_ylabel("Avg Active Power (kW)")
ax.set_xticks(range(24)); ax.legend()
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/hourly_profile.png", dpi=150)
plt.close()

print(f"  Saved charts to {CHART_DIR}/")

# ── BUSINESS INSIGHT SUMMARY ─────────────────────────────────────────────────
conn.close()
peak_month    = monthly.loc[monthly["total_kwh"].idxmax()]
low_month     = monthly.loc[monthly["total_kwh"].idxmin()]
peak_hour     = hourly.loc[hourly["avg_power_kw"].idxmax()]
total_anomalies = df["is_anomaly"].sum()
pct_anomaly   = 100 * total_anomalies / len(df)
latest_year   = submeter.iloc[-1]

print("""
=======================================================
         BUSINESS INSIGHT SUMMARY
=======================================================""")
print(f"  Dataset span       : {df['datetime'].min().date()} → {df['datetime'].max().date()}")
print(f"  Total readings     : {len(df):,} (minute-level)")
print(f"  Total energy used  : {df['energy_kwh'].sum():,.0f} kWh")
print()
print(f"  Peak consumption month : {int(peak_month['year'])}-{int(peak_month['month']):02d}  ({peak_month['total_kwh']:,.0f} kWh)")
print(f"  Lowest consumption     : {int(low_month['year'])}-{int(low_month['month']):02d}  ({low_month['total_kwh']:,.0f} kWh)")
print(f"  Seasonal swing         : {peak_month['total_kwh'] - low_month['total_kwh']:,.0f} kWh difference")
print()
print(f"  Peak demand hour   : {int(peak_hour['hour']):02d}:00  ({peak_hour['avg_power_kw']:.3f} kW avg)")
print(f"  Anomalous readings : {total_anomalies:,}  ({pct_anomaly:.2f}% of all data)")
print()
print(f"  Annual sub-meter breakdown (latest year {int(latest_year['year'])}):")
print(f"    Kitchen (Sub1)  : {latest_year['kitchen_kwh']:>8.1f} kWh")
print(f"    Laundry (Sub2)  : {latest_year['laundry_kwh']:>8.1f} kWh")
print(f"    HVAC    (Sub3)  : {latest_year['hvac_kwh']:>8.1f} kWh")
print(f"    Untracked       : {latest_year['untracked_kwh']:>8.1f} kWh  ← lighting, misc devices")
print()
print("  RECOMMENDATIONS:")
print(f"  1. Shift high-demand appliances away from {int(peak_hour['hour']):02d}:00 peak hour")
print(f"     to reduce grid stress and take advantage of off-peak tariffs.")
print(f"  2. {total_anomalies:,} anomalous readings detected — review for faulty appliances")
print(f"     or energy theft if this were a utility deployment.")
print(f"  3. Untracked consumption is significant — install sub-meters on")
print(f"     lighting circuits to identify further savings.")
print("=======================================================")
print(f"\n  Database : {DB_FILE}")
print(f"  Charts   : {CHART_DIR}/monthly_consumption.png")
print(f"             {CHART_DIR}/hourly_profile.png")