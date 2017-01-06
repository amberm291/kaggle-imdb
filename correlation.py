import csv
import scipy.stats as sci

input_file = csv.DictReader(open("movie_metadata.csv"))
arr1 =[]
arr2 =[]

def correlate(field1,field2):
    for row in input_file:
        if row[field1] == '' or row [field2] == '': continue
        arr1.append(float(row[field1]))
        arr2.append(float(row[field2]))
    print(sci.pearsonr(arr1,arr2))

correlate("num_critic_for_reviews","imdb_score")