# lab.py


import os
import io
from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def trick_me():
    df = pd.DataFrame(columns=['Name', 'Name', 'Age'], index=range(5))
    df['Name'] = 1
    df['Age'] = 2
    df.to_csv('tricky_1.csv', index=False)
    tricky_2 = pd.read_csv('tricky_1.csv')
    # answer should be 3
    return 3


def trick_bool():
    return [4, 10, 13]


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def population_stats(df):

    output = pd.DataFrame(index=df.columns, columns=['num_nonnull', 'prop_nonnull', 'num_distinct', 'prop_distinct'])
    output['num_nonnull'] = df.notna().sum()
    output['prop_nonnull'] = df.notna().sum() / len(df)
    output['num_distinct'] = df.nunique()
    output['prop_distinct'] = df.nunique() / len(df.dropna())

    return output


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def most_common(df, N=10):

    new_df = pd.DataFrame(index=range(N)) 
    for col in df.columns:
        out = df[col].value_counts()
        column_counts = out.iloc[:N] 
        if len(out) < N: 
            padding = N - len(out) 
            padding_series = pd.Series([np.nan] * padding, index=[np.nan] * padding)
            column_counts = pd.concat([column_counts, padding_series])

        new_df[f'{col}_values'] = column_counts.index
        new_df[f'{col}_counts'] = column_counts.values


    return new_df


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def super_hero_powers(powers):
    # part 1
    powers1 = powers.set_index('hero_names')
    powers1['num_powers'] = powers1.sum(axis=1)
    powers1 = powers1.sort_values('num_powers', ascending = False).reset_index()
    name_greatest_num_sp = powers1.iloc[0]['hero_names']
    
    # part 2
    powers2 = powers[powers['Flight'] == True].set_index('hero_names').drop(columns=['Flight']).sum()
    fly_most_comm = powers2.sort_values().idxmax()

    # part 3
    powers3 = powers.set_index('hero_names')
    powers3['num_powers'] = powers3.sum(axis=1)
    powers3 = powers3.sort_values('num_powers', ascending = False)
    powers3 = powers3[powers3['num_powers'] == 1].drop(columns=['num_powers']).sum()
    most_comm_one = powers3.sort_values().idxmax()

    return [name_greatest_num_sp, fly_most_comm, most_comm_one]


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def clean_heroes(heroes):
    return heroes.replace('-', np.nan).replace(-99, np.nan) # numbers less than 0 with np.nan


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def super_hero_stats():
    return ['Fin Fang Foom', 'George Lucas', 'bad', 'Marvel Comics', 'NBC - Heroes', 'Groot']


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def clean_universities(df):
    df_copy = df.copy()

    df_copy['institution'] = df_copy['institution'].replace('\n', ', ', regex=True)
    
    df_copy['broad_impact'] = df_copy['broad_impact'].transform(int)

    df_copy[['nation', 'national_rank_cleaned']] = df_copy['national_rank'].str.split(', ', expand=True)
    df_copy['nation'] = df_copy['nation'].replace({'USA': 'United States'})
    df_copy['nation'] = df_copy['nation'].replace({'Czechia': 'Czech Republic'})
    df_copy['nation'] = df_copy['nation'].replace({'UK': 'United Kingdom'})
    df_copy['national_rank_cleaned'] = df_copy['national_rank_cleaned'].transform(int)
    df_copy = df_copy.drop('national_rank', axis = 1)

    df_copy['is_r1_public'] = ((df_copy['control'].str.contains('Public', na=False)) & df_copy['city'].notnull() & df_copy['state'].notnull())
    
    return df_copy

def university_info(cleaned):
    cleaned['state'].value_counts() >= 3
    state_counts = cleaned['state'].value_counts()
    states_to_keep = state_counts[state_counts >= 3].index
    filtered_df = cleaned[cleaned['state'].isin(states_to_keep)]
    lowest_mean_3_or_more = filtered_df.groupby('state')['score'].mean().sort_values().index[0]

    top_100_school = cleaned[cleaned['world_rank'] <= 100]
    top_100_faculty = top_100_school[top_100_school['quality_of_faculty'] <= 100]
    prop_top_100 = top_100_faculty.shape[0] / top_100_school.shape[0]

    state_pivot_table = cleaned.pivot_table(index='state', columns='is_r1_public', fill_value=0, aggfunc='size')
    state_pivot_table['total'] = state_pivot_table[False] + state_pivot_table[True]
    num_states_priv = int(((state_pivot_table[False] / state_pivot_table['total']) >= .50).sum())

    lowest_world_highest_nation = cleaned[cleaned['national_rank_cleaned'] == 1].sort_values('world_rank').iloc[-1]['institution']

    return [lowest_mean_3_or_more, prop_top_100, num_states_priv, lowest_world_highest_nation]

