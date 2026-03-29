"""
attrition_analysis.py
=====================
Day 7: Refactored Employee Attrition Analysis Module

Industry:  HR / Recruitment
Format:    Python module (.py) — production-grade, testable functions
Skills:    pandas, numpy, seaborn, matplotlib, modular design, docstrings

Who uses this:
    An HR director running quarterly attrition reports. This module
    can be imported, scheduled, or extended — unlike a notebook,
    every function is independently testable and reusable.

Usage:
    python attrition_analysis.py
    pytest test_attrition.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')


# ── Constants ─────────────────────────────────────────────────
TURNOVER_COST_MULTIPLIER = 1.5   # industry standard: 1.5x annual salary
RISK_SCORE_THRESHOLD     = 5     # minimum score to flag for intervention
OUTPUT_DIR               = 'output'
CHART_FILE               = 'attrition_dashboard.png'


# ══════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the IBM HR attrition CSV file into a pandas DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw employee data with original column names preserved.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the given path.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'Data file not found: {filepath}')

    df = pd.read_csv(filepath)
    print(f'[LOAD] Loaded {len(df):,} rows, {len(df.columns)} columns from {filepath}')
    return df


# ══════════════════════════════════════════════════════════════
# CLEAN
# ══════════════════════════════════════════════════════════════

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and enrich the raw HR attrition DataFrame.

    Steps:
    - Add numeric AttritionFlag (1=Yes, 0=No)
    - Drop zero-variance columns (EmployeeCount, StandardHours, Over18)
    - Add TenureBand, SalaryBand, AgeBand categorical columns
    - Add AnnualSalary column

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from load_data().

    Returns
    -------
    pd.DataFrame
        Cleaned and enriched DataFrame ready for analysis.
    """
    df = df.copy()

    # Binary attrition flag
    df['AttritionFlag'] = (df['Attrition'] == 'Yes').astype(int)

    # Drop zero-variance columns
    zero_variance = ['EmployeeCount', 'StandardHours', 'Over18']
    df = df.drop(columns=[c for c in zero_variance if c in df.columns])

    # Tenure bands
    df['TenureBand'] = pd.cut(
        df['YearsAtCompany'],
        bins=[0, 2, 5, 10, 20, 100],
        labels=['0-2 yrs', '3-5 yrs', '6-10 yrs', '11-20 yrs', '20+ yrs'],
        include_lowest=True
    )

    # Salary bands
    df['SalaryBand'] = pd.cut(
        df['MonthlyIncome'],
        bins=[0, 3000, 6000, 10000, 25000],
        labels=['Low (<$3k)', 'Mid ($3-6k)', 'Upper ($6-10k)', 'High (>$10k)'],
        include_lowest=True
    )

    # Age bands
    df['AgeBand'] = pd.cut(
        df['Age'],
        bins=[18, 25, 35, 45, 60],
        labels=['18-25', '26-35', '36-45', '46-60'],
        include_lowest=True
    )

    # Annual salary
    df['AnnualSalary'] = df['MonthlyIncome'] * 12

    print(f'[CLEAN] Clean shape: {df.shape} | Attrition rate: {df["AttritionFlag"].mean()*100:.1f}%')
    return df


# ══════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS
# ══════════════════════════════════════════════════════════════

def calculate_attrition_rate(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Calculate attrition rate, headcount, and leavers for a given grouping column.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from clean_data().
    group_col : str
        Column name to group by (e.g. 'Department', 'JobRole').

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: attrition_rate (%), left, total.
        Sorted by attrition_rate descending.

    Example
    -------
    >>> result = calculate_attrition_rate(df, 'Department')
    >>> result.columns.tolist()
    ['attrition_rate', 'left', 'total']
    """
    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' not found in DataFrame")

    if 'AttritionFlag' not in df.columns:
        raise ValueError("DataFrame must contain 'AttritionFlag' column. Run clean_data() first.")

    result = (
        df.groupby(group_col)['AttritionFlag']
        .agg(['mean', 'sum', 'count'])
        .rename(columns={'mean': 'attrition_rate', 'sum': 'left', 'count': 'total'})
        .assign(attrition_rate=lambda x: (x['attrition_rate'] * 100).round(1))
        .sort_values('attrition_rate', ascending=False)
    )
    return result


def calculate_turnover_cost(
    df: pd.DataFrame,
    group_col: str,
    multiplier: float = TURNOVER_COST_MULTIPLIER
) -> pd.DataFrame:
    """
    Estimate the financial cost of employee turnover by group.

    Uses the industry-standard formula:
        turnover_cost = annual_salary × multiplier

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from clean_data().
    group_col : str
        Column to group by (e.g. 'Department', 'JobRole').
    multiplier : float
        Cost multiplier as fraction of annual salary. Default is 1.5.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: employees_left, avg_annual_salary,
        total_turnover_cost. Sorted by total_turnover_cost descending.
    """
    if 'AnnualSalary' not in df.columns:
        raise ValueError("DataFrame must contain 'AnnualSalary' column. Run clean_data() first.")

    leavers = df[df['Attrition'] == 'Yes'].copy()
    leavers['TurnoverCost'] = leavers['AnnualSalary'] * multiplier

    result = (
        leavers.groupby(group_col)
        .agg(
            employees_left=('AttritionFlag', 'sum'),
            avg_annual_salary=('AnnualSalary', 'mean'),
            total_turnover_cost=('TurnoverCost', 'sum')
        )
        .round(0)
        .sort_values('total_turnover_cost', ascending=False)
    )
    return result


def score_intervention_risk(df: pd.DataFrame, threshold: int = RISK_SCORE_THRESHOLD) -> pd.DataFrame:
    """
    Score current employees (non-leavers) by attrition risk using
    a rule-based weighted scoring model.

    Scoring rules:
    - Overtime = Yes               → +3 points (strongest predictor)
    - Tenure <= 2 years            → +2 points
    - JobSatisfaction <= 2         → +2 points
    - WorkLifeBalance <= 2         → +2 points
    - MonthlyIncome < $3,000       → +1 point
    - YearsSinceLastPromotion >= 4 → +1 point

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from clean_data().
    threshold : int
        Minimum risk score to flag an employee. Default is 5.

    Returns
    -------
    pd.DataFrame
        Employees with RiskScore >= threshold, sorted by RiskScore desc.
        Contains: EmployeeNumber, Department, JobRole, MonthlyIncome,
        YearsAtCompany, JobSatisfaction, WorkLifeBalance, OverTime, RiskScore.
    """
    current = df[df['Attrition'] == 'No'].copy()

    current['RiskScore'] = (
        (current['OverTime'] == 'Yes').astype(int) * 3 +
        (current['YearsAtCompany'] <= 2).astype(int) * 2 +
        (current['JobSatisfaction'] <= 2).astype(int) * 2 +
        (current['WorkLifeBalance'] <= 2).astype(int) * 2 +
        (current['MonthlyIncome'] < 3000).astype(int) * 1 +
        (current['YearsSinceLastPromotion'] >= 4).astype(int) * 1
    )

    flagged = (
        current[current['RiskScore'] >= threshold]
        [[
            'EmployeeNumber', 'Department', 'JobRole', 'MonthlyIncome',
            'YearsAtCompany', 'JobSatisfaction', 'WorkLifeBalance',
            'OverTime', 'RiskScore'
        ]]
        .sort_values('RiskScore', ascending=False)
        .reset_index(drop=True)
    )

    print(f'[SCORE] {len(flagged)} employees flagged with risk score >= {threshold}')
    return flagged


def compare_satisfaction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare average satisfaction scores between employees who stayed vs left.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from clean_data().

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Stayed, Left, Difference, Impact.
        Index is each satisfaction dimension.
    """
    satisfaction_cols = [
        'JobSatisfaction', 'EnvironmentSatisfaction',
        'WorkLifeBalance', 'RelationshipSatisfaction', 'JobInvolvement'
    ]

    result = df.groupby('Attrition')[satisfaction_cols].mean().round(2).T
    result.columns = ['Stayed', 'Left']
    result['Difference'] = (result['Left'] - result['Stayed']).round(2)
    result['Impact'] = result['Difference'].apply(
        lambda x: 'Lower when leaving' if x < 0 else 'Higher when leaving'
    )
    return result


# ══════════════════════════════════════════════════════════════
# VISUALISE
# ══════════════════════════════════════════════════════════════

def plot_dashboard(df: pd.DataFrame, by_dept, by_overtime,
                   by_tenure, by_salary, by_role, sat_comparison,
                   output_path: str = CHART_FILE) -> None:
    """
    Generate and save a 6-panel attrition analysis dashboard.

    Panels:
    - Attrition by department
    - Attrition: overtime vs no overtime
    - Attrition by tenure band
    - Attrition by salary band
    - Satisfaction scores: stayed vs left
    - Correlation heatmap

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from clean_data().
    by_dept, by_overtime, by_tenure, by_salary, by_role : pd.DataFrame
        Attrition rate DataFrames from calculate_attrition_rate().
    sat_comparison : pd.DataFrame
        Satisfaction comparison from compare_satisfaction().
    output_path : str
        File path to save the chart PNG.
    """
    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        'Employee Attrition Analysis — IBM HR Dataset (1,470 Employees)',
        fontsize=14, fontweight='bold', y=1.01
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
    avg_rate = df['AttritionFlag'].mean() * 100

    # Panel 1 — By department
    ax1 = fig.add_subplot(gs[0, 0])
    dept_colors = [
        '#E24B4A' if v > 15 else '#EF9F27' if v > 10 else '#1D9E75'
        for v in by_dept['attrition_rate']
    ]
    bars = ax1.bar(by_dept.index, by_dept['attrition_rate'], color=dept_colors)
    ax1.axhline(avg_rate, color='gray', linestyle='--', linewidth=1,
                label=f'Avg {avg_rate:.1f}%')
    ax1.set_ylabel('Attrition rate (%)')
    ax1.set_title('Attrition by department')
    ax1.legend(fontsize=8)
    ax1.tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, by_dept['attrition_rate']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Panel 2 — Overtime
    ax2 = fig.add_subplot(gs[0, 1])
    ot_colors = ['#E24B4A', '#1D9E75']
    bars2 = ax2.bar(by_overtime.index, by_overtime['attrition_rate'],
                    color=ot_colors, width=0.4)
    ax2.set_ylabel('Attrition rate (%)')
    ax2.set_title('Attrition: overtime vs no overtime')
    for bar, val in zip(bars2, by_overtime['attrition_rate']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Panel 3 — Tenure band
    ax3 = fig.add_subplot(gs[0, 2])
    tenure_data = by_tenure.reindex(
        ['0-2 yrs', '3-5 yrs', '6-10 yrs', '11-20 yrs', '20+ yrs']
    )
    t_colors = [
        '#E24B4A' if v > 20 else '#EF9F27' if v > 15 else '#1D9E75'
        for v in tenure_data['attrition_rate']
    ]
    ax3.bar(tenure_data.index, tenure_data['attrition_rate'], color=t_colors)
    ax3.set_ylabel('Attrition rate (%)')
    ax3.set_title('Attrition by tenure band')
    ax3.tick_params(axis='x', rotation=20)

    # Panel 4 — Salary band
    ax4 = fig.add_subplot(gs[1, 0])
    sal_data = by_salary.reindex(
        ['Low (<$3k)', 'Mid ($3-6k)', 'Upper ($6-10k)', 'High (>$10k)']
    )
    s_colors = [
        '#E24B4A' if v > 20 else '#EF9F27' if v > 10 else '#1D9E75'
        for v in sal_data['attrition_rate']
    ]
    ax4.bar(sal_data.index, sal_data['attrition_rate'], color=s_colors)
    ax4.set_ylabel('Attrition rate (%)')
    ax4.set_title('Attrition by salary band')
    ax4.tick_params(axis='x', rotation=20)

    # Panel 5 — Satisfaction comparison
    ax5 = fig.add_subplot(gs[1, 1])
    x = np.arange(len(sat_comparison))
    width = 0.35
    ax5.bar(x - width/2, sat_comparison['Stayed'], width,
            label='Stayed', color='#378ADD')
    ax5.bar(x + width/2, sat_comparison['Left'], width,
            label='Left', color='#E24B4A')
    ax5.set_xticks(x)
    ax5.set_xticklabels(
        ['Job Sat.', 'Env Sat.', 'WLB', 'Rel Sat.', 'Involvement'],
        rotation=20, fontsize=8
    )
    ax5.set_ylabel('Avg score (1-4)')
    ax5.set_title('Satisfaction: stayed vs left')
    ax5.legend(fontsize=8)
    ax5.set_ylim(0, 4.5)

    # Panel 6 — Correlation heatmap
    ax6 = fig.add_subplot(gs[2, :])
    numeric_cols = [
        'AttritionFlag', 'Age', 'MonthlyIncome', 'YearsAtCompany',
        'JobSatisfaction', 'WorkLifeBalance', 'EnvironmentSatisfaction',
        'DistanceFromHome', 'NumCompaniesWorked', 'TotalWorkingYears',
        'YearsSinceLastPromotion', 'JobLevel', 'StockOptionLevel'
    ]
    corr = df[numeric_cols].corr().round(2)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, ax=ax6,
        cmap='RdBu_r', center=0, vmin=-1, vmax=1,
        annot=True, fmt='.2f', annot_kws={'size': 7},
        linewidths=0.3
    )
    ax6.set_title('Feature correlation heatmap', fontsize=11)
    ax6.tick_params(axis='x', rotation=30, labelsize=8)
    ax6.tick_params(axis='y', rotation=0, labelsize=8)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'[PLOT] Dashboard saved as {output_path}')
    plt.show()


# ══════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════

def export_results(
    by_dept: pd.DataFrame,
    by_role: pd.DataFrame,
    cost_by_dept: pd.DataFrame,
    sat_comparison: pd.DataFrame,
    intervention_list: pd.DataFrame,
    output_dir: str = OUTPUT_DIR
) -> None:
    """
    Export all analysis results to CSV files in the output directory.

    Parameters
    ----------
    by_dept : pd.DataFrame
        Attrition rates by department.
    by_role : pd.DataFrame
        Attrition rates by job role.
    cost_by_dept : pd.DataFrame
        Turnover cost estimates by department.
    sat_comparison : pd.DataFrame
        Satisfaction score comparison.
    intervention_list : pd.DataFrame
        At-risk employee list.
    output_dir : str
        Directory to write CSV files to.
    """
    os.makedirs(output_dir, exist_ok=True)

    by_dept.to_csv(f'{output_dir}/attrition_by_department.csv')
    by_role.to_csv(f'{output_dir}/attrition_by_role.csv')
    cost_by_dept.to_csv(f'{output_dir}/turnover_cost_by_dept.csv')
    sat_comparison.to_csv(f'{output_dir}/satisfaction_comparison.csv')
    intervention_list.to_csv(f'{output_dir}/intervention_priority_list.csv', index=False)

    print(f'[EXPORT] 5 files written to {output_dir}/')


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    """
    Run the full attrition analysis pipeline end to end.

    Steps:
    1. Load raw data
    2. Clean and enrich
    3. Calculate attrition rates across all dimensions
    4. Estimate turnover costs
    5. Score intervention risk for current employees
    6. Compare satisfaction scores
    7. Plot dashboard
    8. Export results
    9. Print business insight summary
    """
    print('=' * 60)
    print('EMPLOYEE ATTRITION ANALYSIS PIPELINE')
    print('=' * 60)

    # 1. Load
    df_raw = load_data('hr_attrition.csv')

    # 2. Clean
    df = clean_data(df_raw)

    # 3. Attrition rates
    by_dept     = calculate_attrition_rate(df, 'Department')
    by_role     = calculate_attrition_rate(df, 'JobRole')
    by_overtime = calculate_attrition_rate(df, 'OverTime')
    by_tenure   = calculate_attrition_rate(df, 'TenureBand')
    by_salary   = calculate_attrition_rate(df, 'SalaryBand')
    by_travel   = calculate_attrition_rate(df, 'BusinessTravel')

    # 4. Turnover cost
    cost_by_dept = calculate_turnover_cost(df, 'Department')
    total_cost   = cost_by_dept['total_turnover_cost'].sum()

    # 5. Risk scoring
    intervention_list = score_intervention_risk(df, threshold=RISK_SCORE_THRESHOLD)

    # 6. Satisfaction comparison
    sat_comparison = compare_satisfaction(df)

    # 7. Plot
    plot_dashboard(
        df, by_dept, by_overtime,
        by_tenure, by_salary, by_role,
        sat_comparison
    )

    # 8. Export
    export_results(
        by_dept, by_role, cost_by_dept,
        sat_comparison, intervention_list
    )

    # 9. Summary
    overall_rate  = df['AttritionFlag'].mean() * 100
    worst_dept    = by_dept.index[0]
    worst_role    = by_role.index[0]
    ot_yes_rate   = by_overtime.loc['Yes', 'attrition_rate'] if 'Yes' in by_overtime.index else 'N/A'
    ot_no_rate    = by_overtime.loc['No',  'attrition_rate'] if 'No'  in by_overtime.index else 'N/A'

    print()
    print('=' * 60)
    print('BUSINESS INSIGHT SUMMARY')
    print('=' * 60)
    print(f'Total employees analysed:    {len(df):,}')
    print(f'Overall attrition rate:      {overall_rate:.1f}%')
    print(f'Employees who left:          {df["AttritionFlag"].sum()}')
    print(f'Total estimated cost:        ${total_cost:,.0f}')
    print()
    print(f'Highest risk department:     {worst_dept} ({by_dept.iloc[0]["attrition_rate"]}% rate)')
    print(f'Highest risk role:           {worst_role} ({by_role.iloc[0]["attrition_rate"]}% rate)')
    print(f'Overtime attrition rate:     {ot_yes_rate}% (vs {ot_no_rate}% without overtime)')
    print(f'Highest attrition tenure:    {by_tenure.index[0]} band')
    print()
    print(f'Employees flagged:           {len(intervention_list)} need immediate check-in')
    print()
    print('TOP RETENTION RECOMMENDATIONS:')
    print(f'  1. Reduce overtime — {ot_yes_rate}% vs {ot_no_rate}% attrition rate')
    print(f'  2. Focus on {worst_dept} — highest attrition rate')
    print(f'  3. Review low salary band compensation — highest flight risk')
    print(f'  4. {len(intervention_list)} employees flagged — prioritise 1:1 check-ins')
    print('=' * 60)


if __name__ == '__main__':
    main()
