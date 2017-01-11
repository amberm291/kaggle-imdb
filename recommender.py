import numpy as np
from scipy.sparse import lil_matrix, coo_matrix
from numpy.linalg.linalg import norm
from numpy import argsort
from sklearn.preprocessing import normalize
import cPickle as pickle 
import csv

def sparse_list_append(row, col, data, j, idx, key, val):
    if key not in idx: return
    row.append(j)
    col.append(idx[key])
    data.append(val)

def gen_matrix(fname, genre_idx, dir_idx, actor_idx, tag_idx):
    row = []
    col = []
    data = []
    j = 0
    movie_list = []
    rdr = csv.DictReader(open(fname))
    for line in rdr:
        tags = line["plot_keywords"]
        genres = line["genres"]
        director = line["director_name"]
        actor1 = line["actor_1_name"]
        actor2 = line["actor_2_name"]
        actor3 = line["actor_3_name"]
        if '' in [tags,genres,director,actor1,actor2,actor3]: continue
        for tag in tags.split('|'):
            sparse_list_append(row,col,data,j,tag_idx,tag,1.0)

        for genre in genres.split('|'):
            sparse_list_append(row,col,data,j,genre_idx,genre,0.25)

        sparse_list_append(row,col,data,j,dir_idx,director,1.0)
        sparse_list_append(row,col,data,j,actor_idx,actor1,0.25)
        sparse_list_append(row,col,data,j,actor_idx,actor2,0.25)
        sparse_list_append(row,col,data,j,actor_idx,actor3,0.25)

        movie_list.append(line["movie_title"])
        j += 1
    row, col, data = map(np.array, [row, col, data])
    feature = coo_matrix((data, (row, col)), shape=(j, len(tag_idx) + len(genre_idx) + len(dir_idx) + len(actor_idx)), dtype=np.float)
    feature = feature.tocsr()
    #feature = normalize(feature,norm='l2',axis=1)
    return feature, np.array(movie_list) #featureNorm

def create_recos(fname, out_fname, dir_pickle, actor_pickle, tag_pickle, genre_pickle):
    dir_idx = pickle.load(open(dir_pickle))
    actor_idx = pickle.load(open(actor_pickle))
    tag_idx = pickle.load(open(tag_pickle))
    genre_idx = pickle.load(open(genre_pickle))

    feature, movie_list = gen_matrix(fname, genre_idx, dir_idx, actor_idx, tag_idx)
    print feature.shape, len(movie_list)
    score_mat = feature.dot(feature.T)
    outfile = open(out_fname,"w")
    for i in xrange(score_mat.shape[0]):
        movie_title = movie_list[i]
        ind = argsort(score_mat[i,:].toarray())[0]
        scores = list(score_mat[i,:].toarray()[0])
        scores = sorted(scores,reverse=True)
        ind = ind[::-1] 
        movie_recos = list(movie_list[ind])
        zipped = zip(movie_recos,scores)

        top_movies = [x[0] for x in zipped if x[0] != movie_title and x[1]>0.0]
        top_movies = top_movies[:15]
        recos = ",".join(map(str,top_movies))
        outfile.write(movie_title + "," + recos + "\n")
    outfile.close()

if __name__ == "__main__":
    fname = "movie_metadata.csv"
    out_fname = "movie_recos.txt"
    dir_pickle = "save_directors.p"
    actor_pickle = "save_actors.p"
    tag_pickle = "save_plot_keywords.p"
    genre_pickle = "save_genres.p"
    create_recos(fname, out_fname, dir_pickle, actor_pickle, tag_pickle, genre_pickle)



