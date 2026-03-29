"""
test_attrition.py
=================
Day 7: Unit tests for the attrition_analysis module

Run with:
    pytest test_attrition.py -v

Tests cover:
    - Data loading and validation
    - Clean data output shape and columns
    - Attrition rate calculation correctness
    - Turnover cost calculation correctness
    - Risk scoring logic
    - Edge cases and error handling
"""

import pytest
import pandas as pd
import numpy as np
import os

from attrition_analysis import (
    load_data,
    clean_data,
    calculate_attrition_rate,
    calculate_turnover_cost,
    score_intervention_risk,
    compare_satisfaction,
)


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def sample_df_raw():
    """
    Minimal raw DataFrame mimicking the IBM HR dataset structure.
    Contains 10 employees — 3 who left, 7 who stayed.
    """
    return pd.DataFrame({
        'Age':                       [25, 35, 28, 45, 32, 50, 29, 38, 41, 26],
        'Attrition':                 ['Yes', 'No', 'Yes', 'No', 'No',
                                      'No', 'Yes', 'No', 'No', 'No'],
        'Department':                ['Sales', 'R&D', 'Sales', 'HR', 'R&D',
                                      'HR', 'Sales', 'R&D', 'R&D', 'Sales'],
        'JobRole':                   ['Sales Rep', 'Scientist', 'Sales Rep', 'HR',
                                      'Scientist', 'Manager', 'Sales Rep',
                                      'Scientist', 'Director', 'Sales Rep'],
        'MonthlyIncome':             [2500, 8000, 1800, 12000, 5500,
                                      15000, 2200, 7000, 10000, 3000],
        'YearsAtCompany':            [1, 8, 2, 15, 5, 20, 1, 6, 12, 3],
        'YearsSinceLastPromotion':   [0, 2, 1, 5, 3, 7, 0, 4, 6, 1],
        'JobSatisfaction':           [2, 4, 1, 3, 4, 3, 2, 4, 3, 3],
        'WorkLifeBalance':           [1, 3, 2, 4, 3, 4, 1, 3, 4, 3],
        'EnvironmentSatisfaction':   [2, 4, 2, 3, 4, 3, 3, 4, 3, 3],
        'JobInvolvement':            [3, 4, 2, 3, 4, 3, 2, 4, 3, 3],
        'RelationshipSatisfaction':  [2, 3, 2, 4, 3, 4, 2, 3, 4, 3],
        'OverTime':                  ['Yes', 'No', 'Yes', 'No', 'No',
                                      'No', 'Yes', 'No', 'No', 'No'],
        'MaritalStatus':             ['Single', 'Married', 'Single', 'Married',
                                      'Divorced', 'Married', 'Single',
                                      'Married', 'Married', 'Single'],
        'Gender':                    ['Male', 'Female', 'Male', 'Female',
                                      'Male', 'Male', 'Female', 'Male', 'Female', 'Male'],
        'BusinessTravel':            ['Travel_Rarely'] * 10,
        'DistanceFromHome':          [5, 10, 3, 20, 8, 15, 4, 12, 18, 6],
        'Education':                 [3, 4, 2, 5, 3, 4, 3, 4, 5, 2],
        'EducationField':            ['Life Sciences'] * 10,
        'EmployeeCount':             [1] * 10,
        'EmployeeNumber':            list(range(1, 11)),
        'JobLevel':                  [1, 3, 1, 4, 2, 5, 1, 3, 4, 2],
        'NumCompaniesWorked':        [2, 1, 3, 0, 2, 1, 4, 2, 1, 3],
        'Over18':                    ['Y'] * 10,
        'PercentSalaryHike':         [15, 12, 20, 11, 14, 13, 18, 15, 12, 16],
        'PerformanceRating':         [3, 4, 3, 3, 4, 3, 3, 4, 3, 3],
        'StandardHours':             [80] * 10,
        'StockOptionLevel':          [0, 1, 0, 2, 1, 3, 0, 1, 2, 0],
        'TotalWorkingYears':         [3, 12, 4, 20, 8, 25, 2, 10, 18, 5],
        'TrainingTimesLastYear':     [2, 3, 1, 4, 3, 2, 2, 3, 4, 2],
        'HourlyRate':                [50, 70, 45, 90, 65, 95, 48, 72, 88, 55],
        'DailyRate':                 [400, 600, 350, 800, 550, 900, 380, 620, 780, 450],
        'MonthlyRate':               [8000, 15000, 7000, 20000, 12000,
                                      25000, 7500, 14000, 19000, 9000],
        'YearsInCurrentRole':        [1, 5, 2, 10, 4, 12, 1, 5, 8, 2],
        'YearsWithCurrManager':      [1, 4, 1, 8, 3, 10, 0, 4, 7, 2],
    })


@pytest.fixture
def sample_df_clean(sample_df_raw):
    """Clean version of the sample DataFrame."""
    return clean_data(sample_df_raw)


# ── Load tests ────────────────────────────────────────────────

class TestLoadData:

    def test_load_raises_on_missing_file(self):
        """load_data should raise FileNotFoundError for non-existent files."""
        with pytest.raises(FileNotFoundError):
            load_data('nonexistent_file.csv')

    def test_load_real_file_if_exists(self):
        """load_data returns a DataFrame when the real file is present."""
        if os.path.exists('hr_attrition.csv'):
            df = load_data('hr_attrition.csv')
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert 'Attrition' in df.columns


# ── Clean tests ───────────────────────────────────────────────

class TestCleanData:

    def test_attrition_flag_created(self, sample_df_clean):
        """clean_data should add an AttritionFlag column."""
        assert 'AttritionFlag' in sample_df_clean.columns

    def test_attrition_flag_is_binary(self, sample_df_clean):
        """AttritionFlag should only contain 0 and 1."""
        unique_vals = set(sample_df_clean['AttritionFlag'].unique())
        assert unique_vals.issubset({0, 1})

    def test_attrition_flag_matches_attrition_col(self, sample_df_clean):
        """AttritionFlag=1 should correspond exactly to Attrition='Yes'."""
        yes_mask  = sample_df_clean['Attrition'] == 'Yes'
        flag_mask = sample_df_clean['AttritionFlag'] == 1
        assert yes_mask.equals(flag_mask)

    def test_zero_variance_cols_dropped(self, sample_df_clean):
        """EmployeeCount, StandardHours, Over18 should be removed."""
        for col in ['EmployeeCount', 'StandardHours', 'Over18']:
            assert col not in sample_df_clean.columns

    def test_tenure_band_created(self, sample_df_clean):
        """clean_data should create TenureBand column."""
        assert 'TenureBand' in sample_df_clean.columns

    def test_salary_band_created(self, sample_df_clean):
        """clean_data should create SalaryBand column."""
        assert 'SalaryBand' in sample_df_clean.columns

    def test_age_band_created(self, sample_df_clean):
        """clean_data should create AgeBand column."""
        assert 'AgeBand' in sample_df_clean.columns

    def test_annual_salary_calculation(self, sample_df_clean):
        """AnnualSalary should equal MonthlyIncome × 12."""
        expected = sample_df_clean['MonthlyIncome'] * 12
        pd.testing.assert_series_equal(
            sample_df_clean['AnnualSalary'],
            expected,
            check_names=False
        )

    def test_no_extra_rows_added(self, sample_df_raw, sample_df_clean):
        """clean_data should not add or remove any rows."""
        assert len(sample_df_clean) == len(sample_df_raw)

    def test_original_not_modified(self, sample_df_raw):
        """clean_data should not modify the original DataFrame (copy safety)."""
        original_cols = sample_df_raw.columns.tolist()
        clean_data(sample_df_raw)
        assert sample_df_raw.columns.tolist() == original_cols


# ── Attrition rate tests ──────────────────────────────────────

class TestCalculateAttritionRate:

    def test_returns_dataframe(self, sample_df_clean):
        """calculate_attrition_rate should return a DataFrame."""
        result = calculate_attrition_rate(sample_df_clean, 'Department')
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self, sample_df_clean):
        """Result should have attrition_rate, left, total columns."""
        result = calculate_attrition_rate(sample_df_clean, 'Department')
        assert set(result.columns) == {'attrition_rate', 'left', 'total'}

    def test_attrition_rate_is_percentage(self, sample_df_clean):
        """Attrition rate should be between 0 and 100."""
        result = calculate_attrition_rate(sample_df_clean, 'Department')
        assert result['attrition_rate'].between(0, 100).all()

    def test_sorted_descending(self, sample_df_clean):
        """Results should be sorted by attrition_rate descending."""
        result = calculate_attrition_rate(sample_df_clean, 'Department')
        rates = result['attrition_rate'].tolist()
        assert rates == sorted(rates, reverse=True)

    def test_known_attrition_rate(self, sample_df_clean):
        """
        Sales has 3 leavers out of 4 employees = 75% attrition rate.
        (Employees 1,3,7 left — all in Sales. Employee 10 stayed.)
        """
        result = calculate_attrition_rate(sample_df_clean, 'Department')
        sales_rate = result.loc['Sales', 'attrition_rate']
        assert sales_rate == 75.0

    def test_total_leavers_correct(self, sample_df_clean):
        """Total leavers across all groups should equal AttritionFlag sum."""
        result = calculate_attrition_rate(sample_df_clean, 'Department')
        assert result['left'].sum() == sample_df_clean['AttritionFlag'].sum()

    def test_raises_on_missing_column(self, sample_df_clean):
        """Should raise ValueError for a non-existent grouping column."""
        with pytest.raises(ValueError):
            calculate_attrition_rate(sample_df_clean, 'NonExistentColumn')

    def test_raises_without_attrition_flag(self, sample_df_raw):
        """Should raise ValueError if AttritionFlag column is missing."""
        with pytest.raises(ValueError):
            calculate_attrition_rate(sample_df_raw, 'Department')


# ── Turnover cost tests ───────────────────────────────────────

class TestCalculateTurnoverCost:

    def test_returns_dataframe(self, sample_df_clean):
        """calculate_turnover_cost should return a DataFrame."""
        result = calculate_turnover_cost(sample_df_clean, 'Department')
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self, sample_df_clean):
        """Result should have expected cost columns."""
        result = calculate_turnover_cost(sample_df_clean, 'Department')
        assert 'total_turnover_cost' in result.columns
        assert 'employees_left' in result.columns
        assert 'avg_annual_salary' in result.columns

    def test_only_leavers_counted(self, sample_df_clean):
        """Total employees_left should equal number of Attrition=Yes rows."""
        result = calculate_turnover_cost(sample_df_clean, 'Department')
        total_leavers = result['employees_left'].sum()
        expected = (sample_df_clean['Attrition'] == 'Yes').sum()
        assert total_leavers == expected

    def test_cost_is_positive(self, sample_df_clean):
        """All turnover costs should be positive."""
        result = calculate_turnover_cost(sample_df_clean, 'Department')
        assert (result['total_turnover_cost'] > 0).all()

    def test_custom_multiplier(self, sample_df_clean):
        """Cost with multiplier=2.0 should be exactly 2/1.5 times the default."""
        result_default = calculate_turnover_cost(sample_df_clean, 'Department')
        result_custom  = calculate_turnover_cost(sample_df_clean, 'Department', multiplier=2.0)
        ratio = (result_custom['total_turnover_cost'].sum() /
                 result_default['total_turnover_cost'].sum())
        assert abs(ratio - (2.0 / 1.5)) < 0.01

    def test_raises_without_annual_salary(self, sample_df_raw):
        """Should raise ValueError if AnnualSalary column is missing."""
        with pytest.raises(ValueError):
            calculate_turnover_cost(sample_df_raw, 'Department')


# ── Risk scoring tests ────────────────────────────────────────

class TestScoreInterventionRisk:

    def test_returns_dataframe(self, sample_df_clean):
        """score_intervention_risk should return a DataFrame."""
        result = score_intervention_risk(sample_df_clean, threshold=1)
        assert isinstance(result, pd.DataFrame)

    def test_only_current_employees(self, sample_df_clean):
        """Result should only contain employees who have NOT left."""
        result = score_intervention_risk(sample_df_clean, threshold=0)
        employee_numbers = result['EmployeeNumber'].tolist()
        leavers = sample_df_clean[sample_df_clean['Attrition'] == 'Yes']['EmployeeNumber'].tolist()
        for num in employee_numbers:
            assert num not in leavers

    def test_risk_score_column_exists(self, sample_df_clean):
        """Result should contain RiskScore column."""
        result = score_intervention_risk(sample_df_clean, threshold=0)
        assert 'RiskScore' in result.columns

    def test_all_scores_above_threshold(self, sample_df_clean):
        """All returned employees should have RiskScore >= threshold."""
        threshold = 3
        result = score_intervention_risk(sample_df_clean, threshold=threshold)
        assert (result['RiskScore'] >= threshold).all()

    def test_sorted_by_risk_score(self, sample_df_clean):
        """Result should be sorted by RiskScore descending."""
        result = score_intervention_risk(sample_df_clean, threshold=0)
        scores = result['RiskScore'].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_high_threshold_returns_empty(self, sample_df_clean):
        """A very high threshold should return an empty DataFrame."""
        result = score_intervention_risk(sample_df_clean, threshold=99)
        assert len(result) == 0

    def test_overtime_adds_three_points(self, sample_df_clean):
        """
        An employee with only OverTime=Yes (no other risk factors)
        should have RiskScore of exactly 3.
        """
        # Build a minimal employee with only overtime as risk factor
        minimal = sample_df_clean[
            (sample_df_clean['Attrition'] == 'No') &
            (sample_df_clean['OverTime'] == 'Yes') &
            (sample_df_clean['YearsAtCompany'] > 2) &
            (sample_df_clean['JobSatisfaction'] > 2) &
            (sample_df_clean['WorkLifeBalance'] > 2) &
            (sample_df_clean['MonthlyIncome'] >= 3000) &
            (sample_df_clean['YearsSinceLastPromotion'] < 4)
        ]
        if len(minimal) > 0:
            result = score_intervention_risk(minimal, threshold=0)
            assert (result['RiskScore'] == 3).all()

    def test_required_columns_in_output(self, sample_df_clean):
        """Output should contain all required columns."""
        required = [
            'EmployeeNumber', 'Department', 'JobRole', 'MonthlyIncome',
            'YearsAtCompany', 'JobSatisfaction', 'WorkLifeBalance',
            'OverTime', 'RiskScore'
        ]
        result = score_intervention_risk(sample_df_clean, threshold=0)
        for col in required:
            assert col in result.columns


# ── Satisfaction comparison tests ─────────────────────────────

class TestCompareSatisfaction:

    def test_returns_dataframe(self, sample_df_clean):
        """compare_satisfaction should return a DataFrame."""
        result = compare_satisfaction(sample_df_clean)
        assert isinstance(result, pd.DataFrame)

    def test_has_stayed_and_left_columns(self, sample_df_clean):
        """Result should have Stayed and Left columns."""
        result = compare_satisfaction(sample_df_clean)
        assert 'Stayed' in result.columns
        assert 'Left' in result.columns

    def test_has_difference_column(self, sample_df_clean):
        """Result should have Difference column."""
        result = compare_satisfaction(sample_df_clean)
        assert 'Difference' in result.columns

    def test_difference_is_left_minus_stayed(self, sample_df_clean):
        """Difference should equal Left - Stayed for all rows."""
        result = compare_satisfaction(sample_df_clean)
        expected = (result['Left'] - result['Stayed']).round(2)
        pd.testing.assert_series_equal(
            result['Difference'], expected, check_names=False
        )

    def test_scores_in_valid_range(self, sample_df_clean):
        """All satisfaction scores should be between 1 and 4."""
        result = compare_satisfaction(sample_df_clean)
        assert result['Stayed'].between(1, 4).all()
        assert result['Left'].between(1, 4).all()
