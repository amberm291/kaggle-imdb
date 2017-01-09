import csv
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
import cPickle as pickle
from sklearn.preprocessing import normalize
from sklearn.linear_model import LinearRegression, Ridge
import math

def remove_strings(attr_list, genre_idx):
    result_list = []
    genre_list = [0.0]*len(genre_idx)
    target_value = None
    for k in xrange(len(attr_list)):
        if k == 9:
            if attr_list[k] == '': return [],[], None
            genres = attr_list[k].split('|')
            for genre in genres:
                genre_list[genre_idx[genre]] = 1.0

        if k in [8,25]: 
            if attr_list[k] == '': return [],[], None
            else: 
                target_value = float(attr_list[25])
                continue

        try:
            result_list.append(float(attr_list[k]))
        except ValueError, e:
            continue

    return result_list, genre_list, target_value

rdr = csv.reader(open("movie_metadata.csv"))
i = 0
first = True
genre_idx = pickle.load(open('save_genres.p'))

for row in rdr:
    if i == 0:
        i += 1
        continue
    feature_list, genre_list, target_value = remove_strings(row, genre_idx)
    if len(feature_list) < 14: continue
    feature_list, genre_list = map(np.array,[feature_list, genre_list])
    if i == 1:
        features = feature_list.copy()
        genre_features = genre_list.copy()
        targets = [target_value]
    else:
        features = np.vstack((features, feature_list))
        genre_features = np.vstack((genre_features,genre_list))
        targets.append(target_value)
    i += 1

l2_factor = math.sqrt(sum(map(lambda x:x*x,targets)))
targets = map(lambda x:x/l2_factor,targets)
features = normalize(features,norm='l2',axis=0)
#pca = PCA(n_components=9, svd_solver='full')
#features = pca.fit_transform(features)
#print features.shape, sum(pca.explained_variance_ratio_)

genre_features = normalize(genre_features,norm='l2',axis=0)
features = np.hstack((features,genre_features))
print features.shape

targets = np.array(targets)

train_idx = np.random.choice(features.shape[0], size=3000, replace=False)
train_features = features[train_idx,:]
train_targets = targets[train_idx]

test_idx = np.array(list(set(range(features.shape[0])) - set(train_idx)))
test_features = features[test_idx,:]
test_targets = targets[test_idx]

print train_features, train_targets

model = Ridge(alpha=0.1)      #LinearRegression()     #
model.fit(train_features,train_targets)
print model.score(test_features,test_targets)

'''
scaler = preprocessing.MinMaxScaler()
scaler.fit(train_features)
features = scaler.transform(train_features)

variance_ratio = []
for i in xrange(len(feature_list)):
    pca = PCA(n_components=i+1, svd_solver='full')
    pca.fit(features)
    print i+1,sum(pca.explained_variance_ratio_)
    variance_ratio.append(sum(pca.explained_variance_ratio_))

print variance_ratio
'''




