import csv

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

    print(genre_idx)

vector_generator(attr_name="genres")
input_file = csv.DictReader(open("movie_metadata.csv"))
vector_generator(attr_name="plot_keywords")