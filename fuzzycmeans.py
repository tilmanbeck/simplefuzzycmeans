import numpy as np

class FuzzyCMeans():

	def __init__(self, data, c, m, error=0.001, maxiter=1000, seed=None):

		self.__data = data
		self.__c = c
		self.__m = m
		self.__error = error
		self.__maxiter = maxiter
		if seed:
			np.random.seed(42)

	def cluster(self):

		(nr_data_points, data_dim) = self.__data.shape

		memberships = np.matrix(np.random.random((self.__c, nr_data_points)))
		centroids = np.array([np.zeros(data_dim) for i in range(self.__c)])
		iterations = 0
		step = 10000
		while(step > self.__error and iterations < self.__maxiter):
			iterations += 1
			# update the centroids
			for i in range(self.__c):
				multiplier = np.power(memberships[i,:],self.__m).transpose()
				centroids[i] = np.divide(np.sum(np.multiply(self.__data, multiplier), axis=0), np.sum(multiplier))

			# re-calculate memberships
			old = np.copy(memberships)
			for i in range(nr_data_points):
				data_point = self.__data[i]
				for j in range(self.__c):
					tmp = np.linalg.norm(data_point - centroids[j])
					bottom = np.sum([np.divide(tmp, np.linalg.norm(data_point - centroids[h])) for h in range(self.__c)])

					memberships[j,i] = np.divide(1, np.power(bottom, (2/(self.__m-1) ) ) )

			# calculate the step distance
			step = np.linalg.norm(memberships - old)

		return (centroids, memberships, iterations)