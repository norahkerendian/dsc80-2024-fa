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
    result = pd.DataFrame()
    redemp_cols = []

    for col in final_breakdown.columns[1:]:
        col_split = col.split(' ')
        q_num = int(col_split[1])
        
        for q in question_numbers:
            if q == q_num:
                redemp_cols.append(col)
    
    result['PID'] = final_breakdown['PID']
    result['total'] = final_breakdown[redemp_cols].sum(axis=1)
    result['Raw Redemption Score'] = result['total'] / result['total'].max()
    result = result[['PID', 'Raw Redemption Score']]
    return result
    
def combine_grades(grades, raw_redemption_scores):
    new_grades = grades.copy()
    return new_grades.merge(raw_redemption_scores, left_on='PID', right_on='PID')


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def z_score(ser):
    return (ser - np.mean(ser)) / np.std(ser, ddof=0)
    
def add_post_redemption(grades_combined):

    def total_score_df(assignment_key):
        assignments = get_assignment_names(grades_combined)
        list_grades = []

        for item in assignments[assignment_key]:
            grades_combined[item] = grades_combined[item].fillna(0)
            item_grade = grades_combined[item] / grades_combined[f'{item} - Max Points']
            list_grades.append(item_grade)

        item_df = pd.DataFrame(data = list_grades, index=assignments[assignment_key]).T
        item_df = item_df.assign(total = item_df.sum(axis = 1) / item_df.shape[1])
        return item_df
    
    grades_copy = grades_combined.copy()

    midterm_grades_final = total_score_df('midterm')['total']

    grades_copy['Midterm Score Pre-Redemption'] = midterm_grades_final

    raw_redemp_z = z_score(grades_copy['Raw Redemption Score'])
    midterm_z = z_score(grades_copy['Midterm Score Pre-Redemption'])

    comp = raw_redemp_z > midterm_z

    post_redemp_score = []
    for ind, boolean in enumerate(comp):
        if boolean == True: 
            # redemption z-score * class' midterm SD + class' midterm mean
            replacement = raw_redemp_z.iloc[ind] * np.std(midterm_grades_final, ddof=0) + np.mean(midterm_grades_final)
            if replacement > 1:
                replacement = 1.0
        else:
            replacement = grades_copy['Midterm Score Pre-Redemption'].iloc[ind]
        
        post_redemp_score.append(replacement)
    
    grades_copy['Midterm Score Post-Redemption'] = post_redemp_score
    
    return grades_copy


# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------
    
def total_points_post_redemption(grades_combined):
    df = add_post_redemption(grades_combined)
    no_midterm = total_points(df) - (df['Midterm Score Pre-Redemption'] * 0.15)
    new_midterm_added = no_midterm + (df['Midterm Score Post-Redemption'] * 0.15)

    return new_midterm_added
        
def proportion_improved(grades_combined):
    pre_redemp = total_points(grades_combined)
    post_redemp = total_points_post_redemption(grades_combined)

    pre_redemp_grade = final_grades(pre_redemp)
    post_redemp_grade = final_grades(post_redemp)

    comp = pre_redemp_grade > post_redemp_grade
    proportion = comp.sum() / len(comp)
    
    return proportion
    


# ---------------------------------------------------------------------
# QUESTION 11
# ---------------------------------------------------------------------


def section_most_improved(grades_analysis):
    
    dictionary = {}
    for section in grades_analysis['Section'].unique():
        df_section = grades_analysis[grades_analysis['Section'] == section].copy()
        ser = df_section['Letter Grade Post-Redemption'] < df_section['Letter Grade Pre-Redemption']
        dictionary[section] = ser.sum() / len(ser)

    return max(dictionary, key=dictionary.get) 
    
def top_sections(grades_analysis, t, n):

    def total_score_df(assignment_key):
        assignments = get_assignment_names(grades_analysis)
        list_grades = []

        for item in assignments[assignment_key]:
            grades_analysis[item] = grades_analysis[item].fillna(0)
            item_grade = grades_analysis[item] / grades_analysis[f'{item} - Max Points']
            list_grades.append(item_grade)

        item_df = pd.DataFrame(data = list_grades, index=assignments[assignment_key]).T
        item_df = item_df.assign(total = item_df.sum(axis = 1) / item_df.shape[1])
        item_df['Section'] = grades_analysis['Section']
        return item_df

    df = total_score_df('final')
    output = np.array([])
    for section in grades_analysis['Section'].unique():
        df_section = df[df['Section'] == section].copy()
        ser_t_comp = df_section['total'] >= t
        if sum(ser_t_comp) >= n:
            output = np.append(section, output)
    output = np.sort(output)
    return output


# ---------------------------------------------------------------------
# QUESTION 12
# ---------------------------------------------------------------------


def rank_by_section(grades_analysis):
    df = grades_analysis.copy()
    rank = grades_analysis.groupby('Section')['Total Points Post-Redemption'].transform(lambda x: x.size - x.argsort().argsort())
    df['rank'] = rank
    return df.pivot(index='rank', columns='Section', values='PID').fillna('')







# ---------------------------------------------------------------------
# QUESTION 13
# ---------------------------------------------------------------------


def letter_grade_heat_map(grades_analysis):
    normalized_grades = (grades_analysis.groupby('Section')['Letter Grade Post-Redemption'].value_counts() / grades_analysis.groupby('Section')['Letter Grade Post-Redemption'].count()).to_frame()
    normalized_grades = normalized_grades.pivot_table(index='Section', columns='Letter Grade Post-Redemption', values=0).fillna(0)
    fig = px.imshow(normalized_grades.T, title='Distribution of Letter Grades by Section', color_continuous_scale='algae')
    return fig
