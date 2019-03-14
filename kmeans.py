import numpy as np
import matplotlib.pyplot as plt

class kMeans(object):

    def __init__(self, k=1, random_state=1):
        self.k = k
        self.random_state = random_state

    def euclidean_distance(self, a, b):
        return np.dot(a - b, a-b)

    def fit(self, X):
        centroids = X[np.random.choice(X.shape[0], self.k, replace=False), :]
        last_iteration_centroids = None
        cluster_assignment = []
        while not np.array_equal(centroids, last_iteration_centroids):
            cluster_assignment = []
            for p in X:
                cluster_assignment.append(np.argmin([self.euclidean_distance(p, c) for c in centroids]))

            new_centroids = []
            for i in range(self.k):
                new_centroids.append(X[np.array(cluster_assignment) == i].mean(axis=0))

            last_iteration_centroids = centroids
            centroids = new_centroids

        for i in range(self.k):
            print("Cluster " +  str(i+1) )
            print(X[np.array(cluster_assignment) == i])
        return cluster_assignment


X = np.array([[2,4],[2,6], [2,8], [10,4], [10,6], [10,8], [900,20], [900,18], [2000,3], [2000,8]])
# X = np.array([])
k = 2
#k = 0

if k<1 or X.size == 0:
    print("Enter Valid Input")
else:
    kmeans = kMeans(k)
    cluster_assignment = kmeans.fit(X)
    plt.scatter(X[:,0],X[:,1], c=cluster_assignment)
    plt.show()

# b. Testing for boundary cases:
#    If k = 0, the code prints "Enter Valid Input"
#    If X is empty, the code prints "Enter Valid Input"

# c. The disadvantages of Kmeans algorithm are:
#           The clusters formed depend on the initial assignment of the centroids.
#           It doesn't work very well on clusters of differing sizes, densities and non-globular shapes.
#           It has problems when the data contains outliers.





