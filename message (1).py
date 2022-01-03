import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lsh import LSH
from cv2 import cv2 as cv
import os
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def create_vect(img):
    red_histo = cv.calcHist([img], [0], None, [256], [0, 256]).flatten()
    green_histo = cv.calcHist([img], [1], None, [256], [0, 256]).flatten()
    blue_histo = cv.calcHist([img], [2], None, [256], [0, 256]).flatten()

    cv.normalize(red_histo, red_histo)
    cv.normalize(green_histo, green_histo)
    cv.normalize(blue_histo, blue_histo)

    img_histo = np.append(red_histo, green_histo)
    img_histo = np.append(img_histo, blue_histo)
    return img_histo


def init_data(path):

    data = np.array([])
    data_names = np.array([])

    for root, dirs, files in os.walk(path):
        for file in files:
            data = np.append(data, (create_vect(cv.imread(path + file, cv.IMREAD_COLOR))))
            data_names = np.append(data_names, file)
        return data.reshape(len(files), 768), data_names.reshape(len(files))


def init_vectors():
    if os.path.isfile('Data768/dataset.npy') and os.path.isfile('Data768/dataset_names.npy'):
        dataset = np.load('Data768/dataset.npy')
        dataset_names = np.load('Data768/dataset_names.npy')
    else:
        dataset, dataset_names = init_data('./Flickr8k/images/')
        np.save('Data768/dataset.npy', dataset)
        np.save('Data768/dataset_names.npy', dataset_names)

    if os.path.isfile('Data768/queries.npy') and os.path.isfile('Data768/queries_names.npy'):
        queries = np.load('Data768/queries.npy')
        queries_names = np.load('Data768/queries_names.npy')
    else:
        queries, queries_names = init_data('./Flickr8k/queries/')
        np.save('Data768/queries.npy', queries)
        np.save('Data768/queries_names.npy', queries_names)

    return dataset, dataset_names, queries, queries_names


def compute_distance_intersection(h1, h2):
    result = 0
    for i in range(len(h1)):
        result += min(h1[i], h2[i])
    return result


def compute_distance_chi2(h1, h2):
    result = 0
    for i in range(len(h1)):
        result += (h1[i] - h2[i])**2/(h1[i] + h2[i])
    return result


def compute_distance_manhattan(data, query):
    result = np.array([])
    for vector in data:
        result = np.append(result, sum(abs(vector - query)))
    return result


def compute_distance_euclide(v, u):
    return np.linalg.norm(v - u)


def compute_distance_tchebychev(v, u):
    return max(abs(v - u))


def knn_search(dataset, query, k):
    nearests = {}

    for index, vector in enumerate(dataset):
        distance = compute_distance_euclide(vector, query)
        if len(nearests) < k:
            nearests[str(index)] = distance
        elif distance < float(max(nearests.values())):
            max_key = max(nearests, key=nearests.get)
            del nearests[max_key]
            nearests[str(index)] = distance
    return nearests


def format_dict(dict):

    keys = np.array([])
    values = np.array([])

    for key, value in dict.items():
        keys = np.append(keys, int(key))
        values = np.append(values, value)

    return [keys, values]


def compute_precision(real_neighbours, finds):
    correct_find = 0
    for find in finds:
        if find in real_neighbours:
            correct_find += 1
    return correct_find/len(finds)


dataset, dataset_names, queries, queries_names = init_vectors()

X = np.array(dataset)
X = numpy.vstack((X, queries))

pca = PCA()
pca.fit(X)

n_components = 0

# On cherche la valeur n_components  pour laquelle la variance expliquée est plus grande que 0.95
for i in range(len(pca.explained_variance_ratio_)):
    n_components = i + 1
    if np.sum(pca.explained_variance_ratio_[:n_components]) > 0.99:
        break

print(n_components)

pca = PCA(n_components=n_components)
X_reduced = pca.fit_transform(X)

X_red_dataset = X_reduced[:len(dataset)]
X_red_queries = X_reduced[len(dataset):]

k = 1

query_index = 0

"""
results = knn_search(Z_red_dataset, Z_red_queries[query_index], k)

print("Avec la distance euclidienne, les %d plus proches voisins de" % k, queries_names[query_index], "sont:")
for key, values in results.items():
    print(dataset_names[int(key)], ", distance :", values)"""

precision = 0

for index, query in enumerate(queries):

    results_base = knn_search(dataset, query, k)
    results_reduced = knn_search(X_red_dataset, X_red_queries[index], k)

    for key, value in results_reduced.items():
        if key in results_base.keys():
            precision += 1

print(precision/len(queries))
