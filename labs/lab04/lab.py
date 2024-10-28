# lab.py


import pandas as pd
import numpy as np
import io
from pathlib import Path
import os


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def prime_time_logins(login):
    login_copy = login.copy()
    login_copy['Time'] = pd.to_datetime(login_copy['Time'])
    output = login_copy[(login_copy['Time'].dt.hour >= 16) & (login_copy['Time'].dt.hour < 20)].groupby('Login Id').count()
    all_users = login['Login Id'].unique()
    output = output.reindex(all_users, fill_value=0)
    return output


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def count_frequency(login):
    login_copy = login.copy()
    login_copy['Time'] = pd.to_datetime(login_copy['Time'])
    today = pd.Timestamp('2024-01-31 23:59:00')

    # number of times they have logged in
    log_in_count = login_copy.groupby('Login Id').count().sort_index()
    # how many days they have been a member
    total_days_member = login_copy.groupby('Login Id').agg(lambda x: (today - x.min()).days).sort_index()

    return log_in_count['Time'] / total_days_member['Time']


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def cookies_null_hypothesis():
    return [1, 2]
                         
def cookies_p_value(N):
    burnt_obs = 15
    total_obs = 250
    prop_obs = burnt_obs / total_obs

    simulated = np.random.choice([0, 1], size=(N, total_obs), p=[0.96, 0.04])
    simulated_props = simulated.sum(axis=1) / total_obs

    # simulations = []

    # for _ in range(N):
    #     simulated = np.random.choice([0,1], size=250, p=[0.96, 0.04])
    #     simulated_prop = simulated.sum() / len(simulated)
    #     simulations.append(simulated_prop)

    p_val = np.mean(simulated_props >= prop_obs)
    # >= becuase i'm trying to disprove my null, anything more than the obs is supporting my null. 
    # we are trying to see if this is far too small to be true 
    return float(p_val)


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def car_null_hypothesis():
    return [1,4]

def car_alt_hypothesis():
    return [2,6]

def car_test_statistic():
    return [1,4]

def car_p_value():
    return 4


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def superheroes_test_statistic():
    return [1, 2]
    
def bhbe_col(heroes):

    blue_eyes = heroes['Eye color'].str.lower().str.contains('blue')
    blond_hair = heroes['Hair color'].str.lower().str.contains('blond')

    p_val_cut_off = 0.01

    return blue_eyes & blond_hair

def superheroes_observed_statistic(heroes):
    filtered_bb = heroes[bhbe_col(heroes)]
    filtered_bb_good = filtered_bb[filtered_bb['Alignment'].str.lower() == 'good']
    obsv_prop = filtered_bb_good.shape[0] / filtered_bb.shape[0]
    return obsv_prop

def simulate_bhbe_null(heroes, N):

    total_good_prop = (heroes['Alignment'].str.lower() == 'good').sum() / heroes.shape[0]
    simulations = np.random.choice([1,0], size=(N, bhbe_col(heroes).sum()), p=[total_good_prop, 1-total_good_prop]) 
    simulated_props = simulations.mean(axis=1)

    return simulated_props

def superheroes_p_value(heroes):
    N = 100_000
    test_stats = simulate_bhbe_null(heroes, N)
    obsv_prop = superheroes_observed_statistic(heroes)
    p_val = np.mean(test_stats >= obsv_prop) # < 0.01 -> reject
    output = [float(p_val)]
    if p_val >= 0.01:
        output.append('Fail to reject')
    else:
        output.append('Reject')
    return output


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def diff_of_means(data, col='orange'):
    # can use for-loop
    grouped_factory = data.groupby('Factory')[col].mean()
    abs_diff = np.abs(grouped_factory.loc['Waco'] - grouped_factory.loc['Yorkville'])
    return abs_diff


def simulate_null(data, col='orange'):
    # can use for-loop
    with_shuffled = data.assign(Shuffled_Factory=np.random.permutation(data['Factory']))
    one_instance_grouped = with_shuffled.groupby('Shuffled_Factory')[col].mean()
    one_instance_test_stat = np.abs(one_instance_grouped.loc['Waco'] - one_instance_grouped.loc['Yorkville'])
    return float(one_instance_test_stat)


def color_p_value(data, col='orange'):
    # can use for-loop
    abs_diff_simulated = []

    for _ in range(1000):
        abs_diff_simulated.append(simulate_null(data, col))
    
    p_val = np.mean(abs_diff_simulated >= diff_of_means(data, col))

    return float(p_val)


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def ordered_colors():
    # can use for-loop
    return [('yellow', 0.000), ('orange', 0.039), ('red', 0.213), ('green', 0.467), ('purple', 0.976)]


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------

    
def same_color_distribution():
    # can use for-loop
    return (0.007, 'Reject')


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def perm_vs_hyp():
    # can use for-loop
    return ['P', 'P', 'P', 'H', 'P']
