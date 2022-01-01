import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lsh import LSH
from cv2 import cv2 as cv
import os
import time


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

    if os.path.isfile('dataset.npy') and os.path.isfile('dataset_names.npy'):
        dataset = np.load('dataset.npy')
        dataset_names = np.load('dataset_names.npy')
    else:
        dataset, dataset_names = init_data('./30k/images/')
        np.save('dataset.npy', dataset)
        np.save('dataset_names.npy', dataset_names)

    if os.path.isfile('queries.npy') and os.path.isfile('queries_names.npy'):
        queries = np.load('queries.npy')
        queries_names = np.load('queries_names.npy')
    else:
        queries, queries_names = init_data('./30k/queries/')
        np.save('queries.npy', queries)
        np.save('queries_names.npy', queries_names)

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

k = 1

queries_neighbours = np.array([])

for query in queries:
    result = knn_search(dataset, query, k)
    queries_neighbours = np.append(queries_neighbours, format_dict(result))

queries_neighbours = queries_neighbours.reshape(len(queries), 2, k)

"""lsh = LSH(10, 3, 5)

lsh.fit(dataset)
result = lsh.kneighbors(queries[0], k=5)

print(dataset_names[result[1][0]])"""

w = 1
nb_tables = 2

for nb_projections in range(1, 15):

    lsh = LSH(nb_projections, nb_tables, w)

    start_build = time.monotonic()
    lsh.fit(dataset)
    end_build = time.monotonic()

    print("Pour un LSH avec les paramètres nb_projections=", nb_projections, ", nb_tables=", nb_tables,
          ", w=", w, ", on a un temps de construction de ", end_build-start_build, "s")

    precisions = np.array([])

    start_search = time.monotonic()
    for query_index, query in enumerate(queries):

        result = lsh.kneighbors(query, k)

        precisions = np.append(precisions, compute_precision(queries_neighbours[query_index, 0, :], result[1]))

    end_search = time.monotonic()
    print(np.mean(precisions))

    print("Pour un LSH avec les paramètres nb_projections=", nb_projections, ", nb_tables=", nb_tables,
          ", w=", w, ", on a un temps de recherche de ", end_search - start_search, "s\n")
