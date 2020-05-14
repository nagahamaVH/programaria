import numpy as np
import pandas as pd
import itertools
from imdb.src.utils import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

raw_imdb = pd.read_csv('./imdb/data/movie_metadata.csv')

imdb = raw_imdb.copy()

imdb['aspect_ratio'] = imdb['aspect_ratio'].astype('str')

imdb['categorical_imdb_score'] = pd.cut(
    imdb['imdb_score'], bins=[0, 4, 6, 8, 10], right=True, labels=False) + 1

imdb['categorical_imdb_score'] = imdb['categorical_imdb_score'].astype('str')

# Useless columns
useless_col = [
    'movie_imdb_link', 'movie_title', 'imdb_score', 'plot_keywords', 
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
quantitative = imdb.select_dtypes(include=['float64', 'int64'])

quantitative = quantitative.transform(lambda x: np.log(x + 0.0001))

imdb = imdb.select_dtypes(exclude=['float64', 'int64'])
imdb = imdb.join(quantitative)

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

genres_dummy = imdb['genres'].str.get_dummies()#.add_prefix('genre_')

genre_effect = [
    'Animation', 'Biography', 'Comedy', 'Documentary', 'Drama', 'History',
    'Music', 'Musical', 'Mystery', 'War']

imdb.drop(['genres'], axis=1, inplace=True)
genres_dummy = genres_dummy[genre_effect]
imdb = imdb.join(genres_dummy)

# --- Creating dummy varible of top-k directors 
director_dummy = get_dummies_based_variables('categorical_imdb_score', 
'director_name', imdb, 5)

director_dummy = director_dummy.add_prefix('director_')

imdb.drop(['director_name'], axis=1, inplace=True)
imdb = imdb.join(director_dummy)

# --- Creating dummy varible of top-k actors 
actor_dummy = get_dummies_based_variables('categorical_imdb_score', 
'actor_1_name', imdb, 5)

actor_dummy = actor_dummy.add_prefix('actor_')

imdb.drop(['actor_1_name'], axis=1, inplace=True)
imdb = imdb.join(actor_dummy)

# --- Handle with missing data
frequency = imdb.isnull().sum().sort_values()
percentual = imdb.isnull().sum() / imdb.isnull().count()

missing_data = pd.concat([frequency, percentual], axis=1, 
keys=['frequency', 'percentual']).sort_values('percentual', ascending=False)

# --- Creating dummy varible of aspect_ratio
aspect_dummy = imdb['aspect_ratio'].str.get_dummies().add_prefix('aspect_')

imdb.drop(['aspect_ratio'], axis=1, inplace=True)
imdb = imdb.join(aspect_dummy)

# --- Creating dummy varible of content_rating
content_dummy = imdb['content_rating'].str.get_dummies().add_prefix('content_')

imdb.drop(['content_rating'], axis=1, inplace=True)
imdb = imdb.join(content_dummy)

# --- Creating dummy varible of color
color_dummy = imdb['color'].str.get_dummies().add_prefix('color_')

imdb.drop(['color'], axis=1, inplace=True)
imdb = imdb.join(color_dummy)

# --- Creating dummy varible of language
language_dummy = imdb['language'].str.get_dummies().add_prefix('language_')

imdb.drop(['language'], axis=1, inplace=True)
imdb = imdb.join(language_dummy)

# --- Creating dummy varible of country
country_dummy = imdb['country'].str.get_dummies().add_prefix('country_')

imdb.drop(['country'], axis=1, inplace=True)
imdb = imdb.join(country_dummy)

# --- Quantitative imputation
missing_data = imdb.isnull().any()
missing_columns = missing_data.index[missing_data].tolist()
median_values = imdb[missing_columns].median()

imdb_clean = imdb.copy()
imdb[missing_columns] = imdb[missing_columns].fillna(median_values)

y = imdb['categorical_imdb_score']
x = imdb.drop(['categorical_imdb_score'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    test_size=0.3,
    random_state=42,
    stratify=y)

standardize = StandardScaler()
x_train = standardize.fit_transform(x_train)
x_test = standardize.transform(x_test)