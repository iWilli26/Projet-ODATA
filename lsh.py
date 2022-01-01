import numpy as np
import heapq as hq
import time as tm

from brute_force import knn_search

class _LSHEntry:

	def __init__(self, data_id, signature):
		self._data_id = data_id
		self._signature = signature

	def __eq__(self, other):
		return self._signature == other._signature

class LSH:
	"""LSH index with regular search and multi-probe search.

	Implementation of E2LSH (LSH for Euclidean-norm). The index contains
	nb_tables hash tables, each using a hash function based on
	nb_projections random projections and a quantization step W.

	:param nb_projections: Number of projections in the hash function of
		 each table.
	:param nb_tables: Number of hash tables.
	:param w: Quantization step of the hash functions.
	:param seed: Seed of the random generator for reproducible experiments.
	:type nb_projections: int
	:type nb_tables: int
	:type w: int
	:type seed: int
	"""

	def __init__(self, nb_projections, nb_tables, w, seed=0):
		self._nb_projections = nb_projections
		self._nb_tables = nb_tables
		self._w = w
		# magic prime for effective hashing
		# (cf. Andoni/Indyk's implementation)
		self._HASH_PRIME = (2 << 31) - 5
		self._TABLE_LENGTH = 2 << 16
		np.random.seed(seed)

	def _generate_vector_projections(self, dimension):
		self._vector_projections = np.random.normal(0, 1, (self._nb_tables, self._nb_projections, dimension))
		self._vector_projections = (self._vector_projections.T / np.linalg.norm(self._vector_projections, axis = 2).T).T
		self._bias = np.random.uniform(0, self._w, (self._nb_tables, self._nb_projections))

	def _generate_signature_projections(self):
		self._signature_projections = np.random.uniform(0, np.random.uniform(0, 2 << 32), (self._nb_projections,)).astype(np.int32)

	def _hash_vector(self, x):
		return ((np.dot(self._vector_projections, x) + self._bias) / self._w).astype(np.int32)

	def _hash_vector_signatures(self, q):
		return np.abs((np.dot(q, self._signature_projections).astype(np.int64)%self._HASH_PRIME)%self._TABLE_LENGTH)

	def _build_tables(self):
		self._tables = np.empty((self._nb_tables, self._TABLE_LENGTH), dtype=np.object)
		for i in range(self._nb_tables):
			for j in range(self._TABLE_LENGTH):
				self._tables[i,j] = []

	def fit(self, data):
		"""Builds the LSH index from the data

		:param data: The data as a (n,d)-shaped Numpy array (n =
			dataset size, d = data dimension).
		:type data: (n,d)-shaped Numpy array.
		"""
		self._data = data
		n, d = data.shape

		# generate random stuff and build tables
		self._generate_vector_projections(d)
		self._generate_signature_projections()
		self._build_tables()

		# populate hash tables with data
		for i, d in enumerate(data):
			signatures = self._hash_vector(d)
			hashs = self._hash_vector_signatures(signatures)
			for t, bucket in enumerate(self._tables[np.arange(len(self._tables)), hashs]):
				bucket.append((i, signatures[t]))

	def kneighbors(self, query, k=1):
		""" Search for the k-NN of query q in the index.

		:param query: the query (or probe) to search.
		:param k: the number of nearest neighbors to return
		:type query: (d,)-shaped Numpy array
		:type k: int

		:return: (matches, distances, nb_matches) with nb_matches the
			size of the short-list on which brute-force search is
			performed, matches the indices of the k-NN in the
			data matrix, and distances the values of the distances
			of the matches to the query.
		:rtype: (int list, float list, int)
		"""
		t1 = tm.time()
		signatures = self._hash_vector(query)
		hashs = self._hash_vector_signatures(signatures)

		# Get matches from buckets
		matches = set()
		t1 = tm.time()
		for signature, bucket in zip(signatures, self._tables[np.arange(len(self._tables)), hashs]):
			for e in bucket:
#				if (e[1] == signature).all():
				matches.add(e[0])

		# brute-force search over matches
		matches = np.array(list(matches))
		if len(matches) > 0:
			t1 = tm.time()
			m, distances = knn_search(self._data[matches], query, k=k, dist="L2")
		else:
			m = []
			distances = []
		return distances, matches[m], len(matches)


