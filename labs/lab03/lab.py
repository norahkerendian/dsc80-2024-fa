# lab.py


import os
import io
from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def read_linkedin_survey(dirname):
    # can use for-loop
    path = Path(dirname)
    output = pd.DataFrame(columns=['first name', 'last name', 'current company', 'job title', 'email', 'university'])

    for csv in path.iterdir():
        df = pd.read_csv(csv)
        df.columns = df.columns.str.lower().str.replace('_',' ')
        output = pd.concat([output, df], ignore_index=True)

    return output
    


def com_stats(df):
    # - The proportion of people who went to a university with the string `'Ohio'` in its name that have the string `'Programmer'` somewhere in their job title.
    # out of all the people, how many have 'ohio' in the university AND 'programmer' in the job title
    prop_ohio_programmer = df[df['university'].str.contains('Ohio') & df['job title'].str.contains('Programmer')].shape[0] / df.shape[0]

    # - The number of job titles that **end** with the exact string `'Engineer'`. Note that we're asking for the number of job titles, **not** the number of people!
    ser = df['job title'].str.endswith('Engineer')
        # this is replacing the nan values with a False
    ser2 = ser == True
    num_engineer = df['job title'][ser2].nunique()

    # - The job title that has the longest name (there are no ties).
    longest_name = df.iloc[df['job title'].str.len().idxmax()]['job title']

    # - The number of people who have the word `'manager'` in their job title, uppercase or lowercase (`'Manager'`, `'manager'`, and `'mANAgeR'` should all count).
    num_manager = df['job title'].str.lower().str.contains('manager').sum()

    return [prop_ohio_programmer, num_engineer, longest_name, num_manager] 



# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def read_student_surveys(dirname):
    # can use for-loop
    path = Path(dirname)
    output = None

    for csv in path.iterdir():
        df = pd.read_csv(csv)
        if output is None:
            output = df
        else:
            output = output.merge(df, on='id')

    return output.set_index('id')


def check_credit(df):
    output = df.copy()
    output['genre'] = output['genre'].replace('(no genres listed)', np.nan)

    extra_credit_whole_class = 0

    if extra_credit_whole_class < 2:
        if output['movie'].count() / output.shape[0] >= 0.9:
            extra_credit_whole_class += 1
        if output['genre'].count() / output.shape[0] >= 0.9:
            extra_credit_whole_class += 1
        if output['animal'].count() / output.shape[0] >= 0.9:
            extra_credit_whole_class += 1
        if output['plant'].count() / output.shape[0] >= 0.9:
            extra_credit_whole_class += 1
        if output['color'].count() / output.shape[0] >= 0.9:
            extra_credit_whole_class += 1

    survey_responses_cols = output.columns.difference(['name'])
    output['count'] = output[survey_responses_cols].count(axis=1)
    output['prop'] = output['count'] / len(survey_responses_cols)
    output['ec'] = extra_credit_whole_class
    output['ec'] = np.where(output['prop'] >= 0.5, extra_credit_whole_class + 5, extra_credit_whole_class)

    return output[['name', 'ec']]


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def most_popular_procedure(pets, procedure_history):
    return pd.merge(pets, procedure_history, on='PetID')['ProcedureType'].value_counts().idxmax()

def pet_name_by_owner(owners, pets):
    df_merged = pd.merge(owners, pets, on='OwnerID')
    df_merged.rename(columns={'Name_x': 'Owner_Name', 'Name_y': 'Pet_Name'}, inplace=True)
    pet_names_by_owner = df_merged.groupby('OwnerID')['Pet_Name'].agg(lambda x: list(x) if len(x) > 1 else x).reset_index()
    pet_names_by_owner_series = pet_names_by_owner.merge(owners, on='OwnerID')[['Pet_Name', 'Name']].set_index('Name')['Pet_Name']
    return pet_names_by_owner_series


def total_cost_per_city(owners, pets, procedure_history, procedure_detail):
    owner_pet_df = pd.merge(owners, pets, on='OwnerID', how='left')
    owner_pet_prodecure_df = pd.merge(owner_pet_df, procedure_history, on='PetID', how='inner')
    completed_df = pd.merge(owner_pet_prodecure_df, procedure_detail, on=['ProcedureType', 'ProcedureSubCode'], how='left')
    total_cost_city = completed_df.groupby('City')['Price'].sum().reset_index()

    all_cities = owners[['City']].drop_duplicates()
    result = pd.merge(all_cities, total_cost_city, on='City', how='left')

    result['Total_Cost'] = result['Price'].fillna(0)
    output = result.set_index('City')['Total_Cost']
    return output


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def average_seller(sales):
    average_sales = sales.pivot_table(
    index='Name',
    values='Total',
    aggfunc='mean'
    )
    average_sales.columns = ['Average Sales']
    return average_sales

def product_name(sales):
    return sales.pivot_table(
        index='Name',
        columns='Product',
        values='Total',
        aggfunc='sum'
    )

def count_product(sales):
    return sales.pivot_table(
        index=['Product', 'Name'],
        columns='Date',
        values='Total',
        aggfunc='sum'
    ).fillna(0)

def total_by_month(sales):
    sale = sales.copy()
    date_format = '%m.%d.%Y'
    sale['Date'] = pd.to_datetime(sale['Date'], format=date_format)
    sale['Month'] = sale['Date'].dt.strftime('%B')
    return sale.pivot_table(
        index=['Name', 'Product'],
        columns='Month',
        values='Total',
        aggfunc='sum'
    ).fillna(0)
