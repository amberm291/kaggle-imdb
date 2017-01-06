import csv
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA

def remove_strings(attr_list):
	result_list = []
	for attr in attr_list:
		try:
			result_list.append(float(attr))
		except ValueError, e:
			continue
	return result_list

rdr = csv.reader(open("movie_metadata.csv"))
i = 0

for row in rdr:
	if i == 0:
		i += 1
		continue
	feature_list = remove_strings(row)
	if len(feature_list) < 16: continue
	feature_list = np.array(feature_list)
	if i == 1:
		features = feature_list.copy()
	else:
		features = np.vstack((features, feature_list))
	i += 1

print features.shape

scaler = preprocessing.MinMaxScaler()
scaler.fit(features)
features = scaler.transform(features)

variance_ratio = []
for i in xrange(len(feature_list)):
	pca = PCA(n_components=i+1, svd_solver='full')
	pca.fit(features)
	print sum(pca.explained_variance_ratio_)
	variance_ratio.append(sum(pca.explained_variance_ratio_))

print variance_ratio





