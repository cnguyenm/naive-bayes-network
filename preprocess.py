
import pandas
import numpy as np


# used for multi-valued attrib
# return a column <bool>, True if value exist in that row
# ex: 'diary' exists in ['diary', 'vegan']
# @param: value <str>, ex: value='casual'
# @param: col_name <str>, ex: 'ambience'
# @return: an array <bool>
def get_row_where(df, col_name, value):
    result = []  # list<bool>

    for row in df[col_name]:

        # row: type<str>
        if row == 'None':
            result.append(False)
            continue

        if value in row:
            result.append(True)
        else:
            result.append(False)

    return result


# replace multi-value attribute with columns (binary value)
# corresponding to their values
# ex: replace 'recommendedFor' with 'breakfast','lunch','brunch'
#
# @param df <DataFrame>
# @return new_df <DataFrame>
def replace_multivalue_attrib(df):

    # ambience
    ambience_unique = ['casual', 'divey', 'trendy', 'romantic', 'classy', 'intimate', 'hipster', 'touristy', 'upscale']
    for value in ambience_unique:
        df[value] = get_row_where(df=df, col_name='ambience', value=value)

    # parking
    parking_unique = ['lot', 'garage', 'street', 'validate', 'valet']
    for value in parking_unique:
        df[value] = get_row_where(df=df, col_name='parking', value=value)

    # dietaryRestrictions
    dietary_unique = ['vegan', 'dairy-free', 'vegetarian', 'gluten-free', 'soy-free', 'halal', 'kosher']
    for value in dietary_unique:
        df[value] = get_row_where(df=df, col_name='dietaryRestrictions', value=value)

    # recommendedFor
    recommend_unique = ['latenight', 'dessert', 'lunch', 'breakfast', 'dinner', 'brunch']
    for value in recommend_unique:
        df[value] = get_row_where(df=df, col_name='recommendedFor', value=value)

    # drop columns
    df = df.drop(
        columns=['ambience', 'parking', 'dietaryRestrictions', 'recommendedFor'])

    return df