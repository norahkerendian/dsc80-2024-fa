# project.py


import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_assignment_names(grades):
    columns = grades.columns
    syllabus = ['lab', 'project', 'midterm', 'final', 'disc', 'checkpoint']
    output_dict = {
        'lab': [], 
        'project': [],
        'midterm': [],
        'final': [],
        'disc': [],
        'checkpoint': []
        }

    for col in columns:
        if len(col.split()) == 1:
            col_lower = col.lower()
            for item in syllabus:
                if item in col_lower:
                    output_dict[item].append(col)
    
    output_dict['project'] = [item for item in output_dict['project'] if len(item.split('_')) == 1]

    return output_dict


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def projects_total(grades):
    project_names = get_assignment_names(grades)['project']
    project_df = pd.DataFrame(index=grades.index)

    for project in project_names:
        grades[project] = grades[project].fillna(0) 
        if f'{project}_free_response' in grades.columns:
            grades[f'{project}_free_response'] = grades[f'{project}_free_response'].fillna(0)
        
        if f'{project}_free_response - Max Points' in grades.columns:
            project_max_scores = grades[f'{project} - Max Points'] + grades[f'{project}_free_response - Max Points']
            project_scores = grades[f'{project}'] + grades[f'{project}_free_response']
        else:
            project_max_scores = grades[f'{project} - Max Points']
            project_scores = grades[f'{project}']

        project_df[project] = project_scores / project_max_scores

    return project_df.sum(axis = 1) / project_df.shape[1]


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def lateness_penalty(col):
    grace_time = pd.Timedelta(hours=2) # 2 hours
    first_penalty = pd.Timedelta(weeks=1)  # 168 hours
    second_penalty= pd.Timedelta(weeks=2) # 336 hours

    col = pd.to_timedelta(col)

    multiplier = pd.Series(index=col.index)

    for i, time in enumerate(col):
        # print(i, lateness)
        if time <= grace_time:
            multiplier[i] = 1.0  
        elif time <= first_penalty:
            multiplier[i] = 0.9  
        elif time <= second_penalty:
            multiplier[i] = 0.7 
        else:
            multiplier[i] = 0.4  

    return multiplier


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def process_labs(grades):
    cols = get_assignment_names(grades)['lab'] # column names
    output = pd.DataFrame(columns = cols, index = grades.index)
    for lab in cols:
        output[lab] = (grades[lab] / grades[f'{lab} - Max Points']) * lateness_penalty(grades[f'{lab} - Lateness (H:M:S)'])
    
    return output


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def lab_total(processed):
    scores = []

    for i in range(processed.shape[0]):
        cur_row = processed.iloc[i]
        cur_row = cur_row.fillna(0)
        cur_row = cur_row.drop(cur_row.idxmin())
        total_lab_score = cur_row.sum() / len(cur_row)
        scores.append(total_lab_score)

    processed['final_score'] = scores
    
    return processed['final_score']


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def total_points(grades):

    def total_score_df(assignment_key):
        assignments = get_assignment_names(grades)
        list_grades = []

        for item in assignments[assignment_key]:
            grades[item] = grades[item].fillna(0)
            item_grade = grades[item] / grades[f'{item} - Max Points']
            list_grades.append(item_grade)

        item_df = pd.DataFrame(data = list_grades, index=assignments[assignment_key]).T
        item_df = item_df.assign(total = item_df.sum(axis = 1) / item_df.shape[1])
        return item_df

    lab_grades_final = lab_total(process_labs(grades)) * 0.20
    project_grades_final = projects_total(grades) * 0.30
    checkpoint_grades_final = total_score_df('checkpoint')['total'] * 0.025
    discussion_grades_final = total_score_df('disc')['total'] * 0.025
    midterm_grades_final = total_score_df('midterm')['total'] * 0.15
    final_grades_final = total_score_df('final')['total'] * 0.30

    course_grade = (lab_grades_final 
                    + project_grades_final 
                    + checkpoint_grades_final 
                    + discussion_grades_final 
                    + midterm_grades_final 
                    + final_grades_final
                    )
    return course_grade


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def final_grades(total):

    letter_grades = []
    for grade in total:
        if grade >= 0.9:
            letter = 'A'
        elif grade >= 0.8:
            letter = 'B'
        elif grade >= 0.7:
            letter = 'C'
        elif grade >= 0.6:
            letter = 'D'
        else:
            letter = 'F'
        letter_grades.append(letter)

    return pd.Series(letter_grades)

def letter_proportions(total):
    input_series = final_grades(total)
    output_series = input_series.value_counts() / len(input_series)
    output_series.sort_values(ascending=False)
    return output_series


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def raw_redemption(final_breakdown, question_numbers):
    ...
    
def combine_grades(grades, raw_redemption_scores):
    ...


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def z_score(ser):
    ...
    
def add_post_redemption(grades_combined):
    ...


# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def total_points_post_redemption(grades_combined):
    ...
        
def proportion_improved(grades_combined):
    ...


# ---------------------------------------------------------------------
# QUESTION 11
# ---------------------------------------------------------------------


def section_most_improved(grades_analysis):
    ...
    
def top_sections(grades_analysis, t, n):
    ...


# ---------------------------------------------------------------------
# QUESTION 12
# ---------------------------------------------------------------------


def rank_by_section(grades_analysis):
    ...







# ---------------------------------------------------------------------
# QUESTION 13
# ---------------------------------------------------------------------


def letter_grade_heat_map(grades_analysis):
    ...
