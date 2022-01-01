import numpy as np
from numpy.random.mtrand import uniform
import matplotlib.pyplot as plt
from cv2 import cv2 as cv
import os
from lsh import LSH
import time
from datetime import timedelta
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Calcule la distance du khi2
def compute_distance_khi(v,u):
    return np.sum(np.power((v-u),2)/(u+v))

# Calcule la distance intersection
def compute_distance_intersection(v,u):
    res=0
    for i in range(len(v)):
        res+=min(v[i],u[i])
    return res

# Calcule la distance de Manhattan
def compute_distance_manhattan(data, query):
    result = np.array([])
    for vector in data:
        result = np.append(result, sum(abs(vector - query)))
    return result

# Calcule la distance euclidienne
def compute_distance_euclide(v, u):
    return np.linalg.norm(v - u)

# Calcule la distance de Tchebychev
def compute_distance_tchebychev(v, u):
    return max(abs(v - u))

# Retourne les k plus proches voisins de query depuis dataset
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


# Retourne un vecteur de taille 768 contenant les 3 vecteurs couleurs RGB
def createVect1(img):
    red_histo = cv.calcHist([img], [0], None, [256], [0, 256]).flatten()
    green_histo = cv.calcHist([img], [1], None, [256], [0, 256]).flatten()
    blue_histo = cv.calcHist([img], [2], None, [256], [0, 256]).flatten()

    cv.normalize(red_histo, red_histo)
    cv.normalize(green_histo, green_histo)
    cv.normalize(blue_histo,blue_histo)

    img_histo = np.append(red_histo, green_histo)
    img_histo = np.append(img_histo, blue_histo)
    return img_histo


# retourne un tableau contenant les vecteurs RGB de chaque image et un second tableau contenant les noms
def init_data(path):
    
    data = np.array([])
    data_names = np.array([])

    for root, dirs, files in os.walk(path):
        for file in files:
            data = np.append(data, (createVect1(cv.imread(path + file, cv.IMREAD_COLOR))))
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


dataset1, dataset_names1, queries1, queries_name1 = init_vectors()


# start_build = time.monotonic()
# a1 = knn_search(dataset1, queries1[0], 5)
# end_build = time.monotonic()
# print("build time 1 :%s" % timedelta(seconds=end_build - start_build))
# for keys,values in a1.items():
#     print(dataset_names1[int(keys)])


'''
lsh = LSH(5,3,5)
start_build = time.monotonic()
lsh.fit(dataset1)
end_build = time.monotonic()
b1 = lsh.kneighbors(queries1[0],64)
print("build time 1 :%s" % timedelta(seconds=end_build - start_build))
for i in range(len(b1[1])):
     print(dataset_names1[b1[1][i]])
'''

# img = cv . imread( "Flickr8k/queries/96978713_775d66a18d.jpg" ,cv .IMREAD_COLOR)
# red_histo = cv.calcHist([img], [0], None, [32], [0, 256], ).flatten()
# green_histo = cv.calcHist([img], [1], None, [32], [0, 256]).flatten()
# blue_histo = cv.calcHist([img], [2], None, [32], [0, 256]).flatten()

# plt.plot (red_histo , color ='r')
# plt.plot (green_histo , color ='g')
# plt.plot (blue_histo , color ='b')
# plt.xlim ([0 ,32])
# plt.show()

print("\n")
def createVect2(img):
    red_histo = cv.calcHist([img], [0], None, [64], [0, 256], ).flatten()
    green_histo = cv.calcHist([img], [1], None, [64], [0, 256]).flatten()
    blue_histo = cv.calcHist([img], [2], None, [64], [0, 256]).flatten()

    cv.normalize(red_histo, red_histo)
    cv.normalize(green_histo, green_histo)
    cv.normalize(blue_histo,blue_histo)

    img_histo = np.append(red_histo, green_histo)
    img_histo = np.append(img_histo, blue_histo)
    return img_histo

# print(np.shape(createVect1(cv.imread("Flickr8k/queries/2393911878_68afe6e6c1.jpg", cv.IMREAD_COLOR))))
# print(np.shape(createVect2(cv.imread("Flickr8k/queries/2393911878_68afe6e6c1.jpg", cv.IMREAD_COLOR))))

def init_data2(path):
    
    data = np.array([])
    data_names = np.array([])

    for root, dirs, files in os.walk(path):
        for file in files:
            data = np.append(data, (createVect2(cv.imread(path + file, cv.IMREAD_COLOR))))
            data_names = np.append(data_names, file)
        return data.reshape(len(files), 768), data_names.reshape(len(files))

def init_vectors2():
    
    if os.path.isfile('Data192/dataset.npy') and os.path.isfile('Data192/dataset_names.npy'):
        dataset = np.load('Data192/dataset.npy')
        dataset_names = np.load('Data192/dataset_names.npy')
    else:
        dataset, dataset_names = init_data2('./Flickr8k/images/')
        np.save('Data192/dataset.npy', dataset)
        np.save('Data192/dataset_names.npy', dataset_names)

    if os.path.isfile('Data192/queries.npy') and os.path.isfile('Data192/queries_names.npy'):
        queries = np.load('Data192/queries.npy')
        queries_names = np.load('Data192/queries_names.npy')
    else:
        queries, queries_names = init_data2('./Flickr8k/queries/')
        np.save('Data192/queries.npy', queries)
        np.save('Data192/queries_names.npy', queries_names)

    return dataset, dataset_names, queries, queries_names
dataset2, dataset_names2, queries2, queries_name2 = init_vectors2()


# start_build = time.monotonic()
# a2 = knn_search(dataset2, queries2[0], 5)
# end_build = time.monotonic()
# print("build time 2 :%s" % timedelta(seconds=end_build - start_build))
# for keys,values in a2.items():
#     print(dataset_names2[int(keys)])


'''
lsh = LSH(5,3,5)
start_build = time.monotonic()
lsh.fit(dataset2)
b2 = lsh.kneighbors(queries2[0],64)
end_build = time.monotonic()
print("build time 2 :%s" % timedelta(seconds=end_build - start_build))
for i in range(len(b2[1])):
     print(dataset_names2[b2[1][i]])
'''
test= np.append(dataset2, queries2)
scaler=StandardScaler()
Z=scaler.fit_transform(test)
acp= PCA(n_components=27)
Z_reduced = acp.fit_transform(Z)
# print(acp.explained_variance_ratio_)
# print(np.cumsum(acp.explained_variance_ratio_)[0:27])    
scaler2=StandardScaler()
Zq=scaler2.fit_transform(queries2)
acp2= PCA(n_components=27)
Zq_reduced = acp.fit_transform(Zq)



start_build = time.monotonic()
a3 = knn_search(Z_reduced, Zq_reduced[0], 10)
end_build = time.monotonic()

'''
print("build time 3 :%s" % timedelta(seconds=end_build - start_build))
for keys,values in a3.items():
    print(dataset_names1[int(keys)])
'''
