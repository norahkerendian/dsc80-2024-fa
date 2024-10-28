# project.py


import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pd.options.plotting.backend = 'plotly'

from IPython.display import display

# DSC 80 preferred styles
pio.templates["dsc80"] = go.layout.Template(
    layout=dict(
        margin=dict(l=30, r=30, t=30, b=30),
        autosize=True,
        width=600,
        height=400,
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        title=dict(x=0.5, xanchor="center"),
    )
)
pio.templates.default = "simple_white+dsc80"
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def clean_loans(loans):
    loan = loans.copy()
    loan['issue_d'] = pd.to_datetime(loan['issue_d'])
    loan['term'] = loan['term'].str.replace(' months', '').astype(int)
    loan['emp_title'] = loan['emp_title'].str.lower()
    loan['emp_title'] = loan['emp_title'].str.strip()
    loan['emp_title'] = loan['emp_title'].apply(lambda x: 'registered nurse' if x == 'rn' else x)
    loan['term_end'] = loan.apply(lambda row: row['issue_d'] + pd.DateOffset(months=row['term']), axis=1)
    return loan


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------



def correlations(df, pairs):
    output_dict = {}

    for col1, col2 in pairs:
        output_dict[f'r_{col1}_{col2}'] = df[col1].corr(df[col2])

    return pd.Series(output_dict)



# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def create_boxplot(loans):
    loan = loans.copy()
    bins = [580, 670, 740, 800, 850]
    loan['binned'] = pd.cut(loan['fico_range_low'], bins=bins, right=False)
    loan['binned'] = loan['binned'].astype(str)

    fig = px.box(
        loan, 
        x="binned", 
        y="int_rate", 
        color="term", 
        category_orders={
            'binned': ['[580, 670)', '[670, 740)','[740, 800)', '[800, 850)'], 
            'term': [36, 60]
            },
        color_discrete_sequence=['blue', 'green']
        )
    fig.update_traces(quartilemethod="linear") 
    fig.update_layout(
        xaxis_title="Credit Score Range",  
        yaxis_title="Interest Rate (%)", 
        legend_title = "Loan Length (Months)",
        title="Interest Rate vs. Credit Score", 
    )
    return fig


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def ps_test(loans, N):
    
    loan = loans.copy()

    actual_means = loan.assign(has_ps=loan['desc'].notna()).groupby('has_ps')['int_rate'].mean()
    actual_difference = actual_means.loc[True] - actual_means.loc[False]

    differences = []
    for _ in range(N):

        with_shuffled = loan.assign(Shuffled_Int_Rates=np.random.permutation(loan['int_rate']))

        group_means = with_shuffled.assign(has_ps=loans['desc'].notna()).groupby('has_ps')['Shuffled_Int_Rates'].mean()

        difference = group_means.loc[True] - group_means.loc[False]
        
        differences.append(difference)
        
    p_val = np.count_nonzero(differences >= actual_difference) / N

    return p_val
    
def missingness_mechanism():
    return 2
    
def argument_for_nmar():
    '''
    Could be considered NMAR since applicants have the option to include a personal statement 
    which we can interpret as self-reporting, the applicant might be embarrassed to disclose 
    why they are taking out a loan.
    '''
    return "Could be considered NMAR since applicants have the option to include a personal statement which we can interpret as self-reporting, the applicant might be embarrassed to disclose why they are taking out a loan."


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def tax_owed(income, brackets):
    total = 0

    for tup in range(len(brackets)-1):
        # this if statement deals with the tax brackets before the one i have to take a portion of
        if income > brackets[tup+1][1]:
            tup_range = brackets[tup+1][1] - brackets[tup][1]
            total += tup_range * brackets[tup][0]

        if income <= brackets[tup+1][1]:
            tup_range = income - brackets[tup][1]
            total += tup_range * brackets[tup][0]
            break
    if income >= brackets[-1][1]:
        tup_range = income - brackets[-1][1]
        total += tup_range * brackets[-1][0]

    return total


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def clean_state_taxes(state_taxes_raw): 
    state_taxes_raw_copy = state_taxes_raw.copy()
    state_taxes_raw_copy = state_taxes_raw_copy.dropna(how="all")
    state_taxes_raw_copy.loc[state_taxes_raw_copy['State'].str.contains(r'[\(\)]', na=False), 'State'] = np.nan
    state_taxes_raw_copy['State'] = state_taxes_raw_copy['State'].ffill()
    state_taxes_raw_copy['Rate'] = state_taxes_raw_copy['Rate'].str.replace('%', '')
    state_taxes_raw_copy['Rate'] = state_taxes_raw_copy['Rate'].str.replace('none', '0.00')
    state_taxes_raw_copy['Rate'] = (state_taxes_raw_copy['Rate'].astype(float) / 100).round(2)
    state_taxes_raw_copy['Lower Limit'] = state_taxes_raw_copy['Lower Limit'].str.replace('$', '').str.replace(',', '').fillna(0).astype(int)
    return state_taxes_raw_copy


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def state_brackets(state_taxes):
    grouped = state_taxes.groupby('State').apply(lambda x: list(zip(x['Rate'] , x['Lower Limit'])))
    formated = grouped.reset_index().rename(columns={0: 'bracket_list'}).set_index('State')
    return formated

    
def combine_loans_and_state_taxes(loans, state_taxes):
    # Start by loading in the JSON file.
    # state_mapping is a dictionary; use it!
    import json
    state_mapping_path = Path('data') / 'state_mapping.json'
    with open(state_mapping_path, 'r') as f:
        state_mapping = json.load(f)
        
        
    # Now it's your turn:
    bracket_df = state_brackets(state_taxes)
    bracket_df = bracket_df.reset_index()
    bracket_df['State'] = bracket_df['State'].replace(state_mapping)

    merged = loans.merge(bracket_df, left_on='addr_state', right_on='State')
<<<<<<< Updated upstream
<<<<<<< Updated upstream
    merged = merged.drop('addr_state', axis=1)
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes

    return merged


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def find_disposable_income(loans_with_state_taxes):
    FEDERAL_BRACKETS = [
     (0.1, 0), 
     (0.12, 11000), 
     (0.22, 44725), 
     (0.24, 95375), 
     (0.32, 182100),
     (0.35, 231251),
     (0.37, 578125)
    ]
    loans_with_state_taxes_copy = loans_with_state_taxes.copy()

    # federal_tax_owed
    federal_tax_owed = []
    for i in range(len(loans_with_state_taxes['annual_inc'])):
        federal_tax_owed.append(tax_owed(loans_with_state_taxes['annual_inc'].iloc[i], FEDERAL_BRACKETS))
    federal_tax_owed = pd.Series(federal_tax_owed)
    loans_with_state_taxes_copy['federal_tax_owed'] = federal_tax_owed

    # state_tax_owed
    state_tax_owed = []
    for i in range(len(loans_with_state_taxes['annual_inc'])):
        state_tax_owed.append(tax_owed(loans_with_state_taxes['annual_inc'].iloc[i], loans_with_state_taxes['bracket_list'].iloc[i]))
    state_tax_owed = pd.Series(state_tax_owed)
    loans_with_state_taxes_copy['state_tax_owed'] = state_tax_owed

    # disposable_income
    loans_with_state_taxes_copy['disposable_income'] = loans_with_state_taxes_copy['annual_inc'] - loans_with_state_taxes_copy['federal_tax_owed'] - loans_with_state_taxes_copy['state_tax_owed']

    return loans_with_state_taxes_copy

# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def aggregate_and_combine(loans, keywords, quantitative_column, categorical_column):
    ...


# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def exists_paradox(loans, keywords, quantitative_column, categorical_column):
    ...
    
def paradox_example(loans):
    return {
        'loans': loans,
        'keywords': [..., ...],
        'quantitative_column': ...,
        'categorical_column': ...
    }
