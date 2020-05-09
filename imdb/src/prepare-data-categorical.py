import numpy as np
import pandas as pd
import itertools
from imdb.src.utils import *

raw_imdb = pd.read_csv('./imdb/data/movie_metadata.csv')

imdb = raw_imdb.copy()

imdb['aspect_ratio'] = imdb['aspect_ratio'].astype('str')

imdb['categorical_imdb_score'] = pd.cut(imdb['imdb_score'], 
bins=[0, 4, 6, 8, 10], right=True, labels=False) + 1

# Useless columns
useless_col = ['movie_imdb_link', 'movie_title', 'imdb_score', 'plot_keywords',
'actor_2_name', 'actor_3_name']

imdb.drop(useless_col, axis=1, inplace=True)

# High correlation variables
correlated_var = ['cast_total_facebook_likes']

imdb.drop(correlated_var, axis=1, inplace=True)

# Creating new variables
imdb['return_investment'] = (imdb['gross'] - imdb['budget']) / imdb['budget']
imdb['return_investment'] = imdb['return_investment'].mask(
    imdb['return_investment'] < 0, 0)

# --- Transforming quantitative variables to log scale
quantitative = imdb.select_dtypes(include=['float64', 'int64']).drop(
    ['categorical_imdb_score'], axis=1)

quantitative = quantitative.transform(lambda x: np.log(x + 0.0001))

imdb = imdb.select_dtypes(exclude=['float64', 'int64'])
imdb = pd.concat([imdb, quantitative], axis=1)

# --- Grouping levels of aspect ratio of screen
for i in range(0, len(imdb)):
    value = imdb.loc[i, 'aspect_ratio']
    if value == 'nan':
        imdb.loc[i, 'aspect_ratio'] = np.nan
    elif value not in ['2.35', '1.85', '1.78', '1.37']:
        imdb.loc[i, 'aspect_ratio'] = 'Other'

# --- Grouping levels of content rating
for i in range(0, len(imdb)):
    value = imdb.loc[i, 'content_rating']
    if value not in ['PG-13', 'R', 'PG'] and str(value) != 'nan':
        imdb.loc[i, 'content_rating'] = 'Other'

# --- Grouping levels of language
for i in range(0, len(imdb)): 
    value = imdb.loc[i, 'language']
    if  value not in ['English'] and str(value) != 'nan':
        imdb.loc[i, 'language'] = 'Other'

# --- Grouping levels of country
for i in range(0, len(imdb)):
    value = imdb.loc[i, 'country']
    if value not in ['USA', 'UK', 'France', 'Canada', 'Germany', 
    'Australia'] and str(value) != 'nan':
        imdb.loc[i, 'country'] = 'Other'

# --- Creating dummy varible of genres 
genres_list = imdb['genres'].str.split('|').tolist()

unique_genres = set(list(itertools.chain.from_iterable(genres_list)))

genres_dummy = imdb['genres'].str.get_dummies().add_prefix('genre_')

imdb.drop(['genres'], axis=1, inplace=True)
imdb = pd.concat([imdb, genres_dummy], axis=1)

# --- Creating dummy varible of top-k directors 
director_dummy = get_dummies_based_variables('categorical_imdb_score', 
'director_name', imdb, 5)

director_dummy = director_dummy.add_prefix('director_')

imdb.drop(['director_name'], axis=1, inplace=True)
imdb = pd.concat([imdb, director_dummy], axis=1)

# --- Creating dummy varible of top-k actors 
actor_dummy = get_dummies_based_variables('categorical_imdb_score', 
'actor_1_name', imdb, 5)

actor_dummy = actor_dummy.add_prefix('actor_')

imdb.drop(['actor_1_name'], axis=1, inplace=True)
imdb = pd.concat([imdb, actor_dummy], axis=1)

# --- Inpute missing data
frequency = imdb.isnull().sum().sort_values()
percentual = imdb.isnull().sum() / imdb.isnull().count()

missing_data = pd.concat([frequency, percentual], axis=1, 
keys=['frequency', 'percentual']).sort_values('percentual', ascending=False)

imdb.columns

#imdb.to_csv('./imdb/data/categorical_imdb.csv', index=False, sep=';')