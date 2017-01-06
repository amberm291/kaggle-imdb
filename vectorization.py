import csv
import numpy as np
import pickle
input_file = csv.DictReader(open("movie_metadata.csv"))

def vector_generator(attr_name):
    genre_idx = {}
    idx = 0
    for row in input_file:
        genre_array = row[attr_name].split("|")
        #print(genre_array)
        #break

        for str in genre_array:
            if str not in genre_idx:
                genre_idx[str] = idx
                idx = idx + 1

    return (genre_idx)

#vector_generator(attr_name="genres")
pickle.dump(vector_generator(attr_name="genres"), open("save_genres.p", "wb"),protocol=2)
input_file = csv.DictReader(open("movie_metadata.csv"))
pickle.dump( vector_generator(attr_name="plot_keywords"), open("save_plot_keywords.p", "wb"),protocol=2)
#vector_generator(attr_name="plot_keywords")
