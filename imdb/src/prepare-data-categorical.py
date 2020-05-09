import numpy as np
import pandas as pd
import itertools

raw_imdb = pd.read_csv('./imdb/data/movie_metadata.csv')

imdb = raw_imdb.copy()

imdb['aspect_ratio'] = imdb['aspect_ratio'].astype('object')

imdb['categorical_imdb_score'] = pd.cut(imdb['imdb_score'], 
bins=[0, 4, 6, 8, 10], right=True, labels=False) + 1

useless_col = ['movie_imdb_link', 'movie_title', 'imdb_score']

imdb.drop(useles_col, axis=1, inplace=True)

# --- Creating dummy varible of genres 
genres_list = imdb['genres'].str.split('|').tolist()

unique_genres = set(list(itertools.chain.from_iterable(genres_list)))

genres_dummy = imdb['genres'].str.get_dummies().add_prefix('genre_')

imdb.drop(['genres'], axis=1, inplace=True)
imdb = pd.concat([imdb, genres_dummy], axis=1)