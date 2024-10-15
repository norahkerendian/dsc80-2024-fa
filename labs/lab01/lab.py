# lab.py


from pathlib import Path
import io
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# QUESTION 0
# ---------------------------------------------------------------------


def consecutive_ints(ints):
    if len(ints) == 0:
        return False

    for k in range(len(ints) - 1):
        diff = abs(ints[k] - ints[k+1])
        if diff == 1:
            return True

    return False


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def median_vs_mean(nums):
    nums.sort() # in-place
    length = len(nums)
    half_way = length // 2
    # sort the list first then determine if the list is even or odd then find the middle number 
    if length % 2 == 0: # even
        list_median = (nums[half_way - 1] + nums[half_way])/2
    else:
        list_median = nums[half_way]

    list_mean = sum(nums) / length

    # print(f'mean = {list_mean}')
    # print(f'median = {list_median}')
    return list_median <= list_mean



# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def n_prefixes(s, n):
    # s[n-1] + s[n-2] + s[n-3] ...
    # there has to be a way to do this recursively 
    if n == 0:
        # so the base case has to be n decreasing to 0
        return ""
    else:
        # s is the new prefix everytime n decreases 
        s = s[:n]
        return s + n_prefixes(s, n-1)
        


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def exploded_numbers(ints, n):
    largest_val = max(ints) + n
    output = []
    for i in ints:
        first = i - n
        end = i + n
        string_num = ' '.join(str(j).zfill(len(str(largest_val))) for j in range(first, end + 1))
        output.append(string_num)
        # str.zfill(max(len(str(largest_val))))
    return output



# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def last_chars(fh):
    s = fh.read()
    s = s.split('\n')
    output = ''
    for line in s:
        if line != '':
            last = line[-1]
            output += last
    return output
        


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def add_root(A):
    # (1, 5, 7) -> (1 + sqrt(0), 5 + sqrt(1), 7 + sqrt(2))
    # i need to make an array of the sqrts and then just add the two arrays together
    length = len(A)
    sqrt_array = np.sqrt(np.array(range(0, length)))
    return A + sqrt_array

def where_square(A):
    # [2,9,16,15] -> [False, True, True, False]
    # sqrt(int) == int but idk if this can be done w/o a loop 
    squared_A = np.sqrt(A)
    squared_A_convert = squared_A.astype(int)
    return squared_A == squared_A_convert




# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def filter_cutoff_loop(matrix, cutoff):

    groups = [[row[i] for row in matrix] for i in range(len(matrix[0]))]
    means = [sum(i)/len(i) for i in groups]
    index_list = []
    for i, num in enumerate(means):
        if num > cutoff:
            index_list.append(i)
 
    new_matrix = []      
    for row in matrix:
        saved_rows = []
        for i in index_list:
            saved_rows.append(row[i])
        new_matrix.append(saved_rows)
    
    return np.array(new_matrix)




# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def filter_cutoff_np(matrix, cutoff):
    means = np.mean(matrix, axis = 0)
    return matrix[:, means > cutoff]



# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def growth_rates(A):
    diff_array = np.diff(A)
    growth = diff_array / A[:-1]
    return np.round(growth,2)
# round((A[i+1]- A[i]) / A[i], 2)


def with_leftover(A):
    # when leftover is greater than or equal to the stock price of the same day, return that day
    # np.cumsum() should be used for the money that is left over. 
    # 20 % stock price to get left over
    # then add the left over with cumsum 
    # but still check after everyday if you have money left over to buy stock. 
    # if total leftover at the end (so the last element of the cumsum array) is less than the last stock day, then return -1

    # remainder = 20%A
    # leftover = np.cumsum(remainder) # this should add up every time
    # for i in range(len(A)):
    #     if A[i] <= leftover[i-1]:
    #         return np.int64(i + 1)
    # return np.int64(-1)

    remainder = 20%A
    leftover = np.cumsum(remainder)

    mask_TF = (A <= leftover)

    first_day_enough = np.where(mask_TF)[0]

    if first_day_enough.size > 0:
        return np.int64(first_day_enough[0])
    else:
        return np.int64(-1)
    
    


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def salary_stats(salary):
    # have a dictionary that you update and then cast it into a series
    output_dic = {}

    # 'num_players': groupby player and then count how many rows there are
    output_dic['num_players'] = salary.groupby('Player').count().shape[0]

    # 'num_teams': groupby team and then count how many rows there are
    output_dic['num_teams'] = salary.groupby('Team').count().shape[0]

    # 'total_salary': just sum up the salary column (if this fails, then maybe you have to groupby player first and then sum up salary)
    output_dic['total_salary'] = int(salary['Salary'].sum())
    # output_dic['total_salary_gr'] = int(salary.groupby('Player').sum()['Salary'].sum())

    # 'highest_salary': find the max salary 
    output_dic['highest_salary'] = salary.sort_values(by='Salary', ascending = False).iloc[0]['Player']

    # 'avg_loss': groupby team and use mean() and then loc the Lakers and round the salary
    output_dic['avg_los'] = float(salary.groupby('Team')['Salary'].mean().loc['Los Angeles Lakers'].round(2))

    # 'fifth_lowest': sort the salary column and find the fifth row and get the name and team and then concatonate in string format. 
    player = salary.sort_values(by = "Salary").iloc[4]['Player']
    team = salary.sort_values(by = "Salary").iloc[4]['Team']
    output_dic['fifth_lowest'] = player + ', ' + team

    # 'duplicates': this can possibl be done with indexing and getting the 1 positon for each name and comparing and there if there is one, return true
    just_last = salary['Player'].str.split().str[1]
    output_dic['duplicates'] = bool(just_last.duplicated().any())

    # 'total_highest': sort the salary column and go to the max value and return it's corresponding team name. then groupby team name and sum the salary.
    team_sal = salary.sort_values(by = "Salary", ascending = False).iloc[0]['Team']
    output_dic['total_highest'] = int(salary.groupby('Team').sum().loc[team_sal]["Salary"])
    return pd.Series(output_dic)

# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def parse_malformed(fp):
    file = open(fp, 'r')
    # , or " are at the end or beginning of each entry
    df_list = []
    cols = True
    for i in file:
        # reads one line at a time

        # replace the "" with nothing
        # then split the lines by the ,
        # then check if each split list is of length 5
        # if it is not of length 5, then loop throigh and remove empty ''
        # then use pd.dataframe to convert it to a dataframe
        # add together the last two columns to get the 'geo' column and format it as needed

        removed_quotes = i.replace('"', '')
        removed_enter = removed_quotes.replace('\n', '')
        split_comma = removed_enter.split(',')
        if len(split_comma) > 6:
            split_comma.remove('')
        split_comma[-2:] = [str(split_comma[-2]) + ',' + str(split_comma[-1])]

        if cols == True:
            split_comma[-1:] = split_comma[-1].split(',')
            cols = False
        else:
            split_comma[2] = float(split_comma[2])
            split_comma[3] = float(split_comma[3])
        df_list.append(split_comma)

    # df_list[0][-1:] = df_list[0][-1].split(',')
    df = pd.DataFrame(data = df_list[1:], columns=df_list[0])
    return df
