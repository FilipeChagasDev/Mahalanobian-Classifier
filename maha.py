import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.cluster import KMeans
from kneed import KneeLocator
from matplotlib import pyplot as plt
from typing import *

euclidean_distance = lambda a,b: np.linalg.norm(a-b, axis=1)

class OneCluster():
    """
    A fake KMeans of just one cluster.
    """
    def __init__(self):
        self.x = None
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, x: np.ndarray):
        self.x = x
        centroid = np.mean(x, axis=0)
        self.cluster_centers_ = np.expand_dims(centroid, axis=0)
        self.labels_ = np.zeros(shape=(x.shape[0],))
        self.inertia_ = np.sum(euclidean_distance(x, np.tile(centroid, reps=(x.shape[0],1))**2), axis=0)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.zeros(shape=(x.shape[0],))

def kmeans_elbow(x: np.ndarray, max_clusters_per_class: int = None, slope_limit: float = 1, persistence: int = 10, verbose: bool = False, chart: bool = False) -> Union[KMeans, OneCluster]:
    """Adjusts KMeans for data by setting the ideal cluster number by the elbow method. 

    Args:
        x (np.ndarray): Data array. Each row is a sample.
        max_clusters_per_class (int, optional): Maximum number of clusters for or kmeans. Defaults to None.
        slope_limit (float, optional): Slope limit (in degrees) for the inertia curve. When this limit is reached, the tests remain only by a defined number of times. Defaults to 1.
        persistence (int, optional): Number of tests to be performed after the slope limit is reached. Defaults to 10.
        verbose (bool, optional): Set as True to view text messages. Defaults to False.
        chart (bool, optional): Define as true to see a line chart of inertia. Defaults to False.

    Returns:
        Union[KMeans, OneCluster]: [description]
    """
    assert x.shape[0] > 1
    assert isinstance(max_clusters_per_class, int) or isinstance(max_clusters_per_class, type(None))
    assert max_clusters_per_class > 2 if isinstance(max_clusters_per_class, int) else True
    assert x.shape[0] >= max_clusters_per_class if isinstance(max_clusters_per_class, int) else True

    if max_clusters_per_class == None:
        max_clusters_per_class = x.shape[0]
    
    print('applying kmeans...') if verbose else None
    kmeans_instances = [OneCluster().fit(x)]
    xinertia = [0]
    yinertia = [kmeans_instances[0].inertia_]
    persistence_counting = 0
    for n in range(2, max_clusters_per_class+1):
        kmeans = KMeans(n).fit(x)
        xinertia.append(xinertia[-1]+1)
        yinertia.append(kmeans.inertia_)

        angular_coeff_0 = -(yinertia[1]-yinertia[0])/(xinertia[1]-xinertia[0])
        angular_coeff_n = -(yinertia[-1]-yinertia[-2])/(xinertia[-1]-xinertia[-2])
        slope = (np.arctan(angular_coeff_n/angular_coeff_0)*180)/np.pi
        
        print(f'{n} clusters. inertia={kmeans.inertia_} slope={slope}') if verbose else None

        if slope <= slope_limit:
            persistence_counting += 1
        else:
            persistence_counting = 0

        if persistence_counting == persistence:
            print(f'stoping here') if verbose else None
            break
    
    kl = KneeLocator(xinertia, yinertia, curve='convex', direction='decreasing')

    if chart:
        plt.plot(xinertia, yinertia, label='Inertia')
        plt.axvline(kl.elbow, ls=':', color='gray', label='Elbow')
        plt.xlabel('Number of clusters')
        plt.legend()
        plt.grid()
        plt.show()
    
    return kmeans_instances[kl.elbow if isinstance(kl.elbow, type(None)) else 0]

def separate_classes(x: np.ndarray, y: np.ndarray) -> List[Tuple[int, np.ndarray]]:
    """Separate samples by classes into a list.

    Args:
        x (np.ndarray): Samples.
        y (np.ndarray): Target labels (classes).

    Returns:
        List[Tuple[int, np.ndarray]]: List in the format [(class, samples),...]
    """
    classes = np.unique(y)
    l = []
    for clss in classes:
        l.append((clss, x[y==clss]))
    return l

class MahalanobianClassifier():
    def __init__(self, max_clusters_per_class: int = None, limit: float = 2):
        """
        Args:
            max_clusters_per_class (int, optional): Maximum cluster number in each class. Defaults to None.
            limit (float, optional): Limit distance so the samples are not classified as None. It can be set to float('inf') if the None classification is not desirable. Defaults to 2.
        """
        self.limit = limit
        self.classes_kmeans = dict()
        self.clusters_params = dict()

    def fit(self, x: np.ndarray, y: np.ndarray):
        """Adjusts the classifier for the dataset.

        Args:
            x (np.ndarray): Array of samples. Each row should be a sample.
            y (np.ndarray): Array of labels. Labels should be integer.

        Returns:
            MahalanobianClassifier: The classifier itself.
        """
        for clss, samples in separate_classes(x,y):
            kminstance = kmeans_elbow(samples, verbose=True)
            samples_clusters = kminstance.predict(samples)
            
            self.clusters_params[clss] = dict()
            for cluster, samples2 in separate_classes(samples, samples_clusters):
                centroid = kminstance.cluster_centers_[int(cluster)]
                self.clusters_params[clss][cluster] = dict()
                self.clusters_params[clss][cluster]['cov'] = np.cov(samples2, rowvar=False)
                self.clusters_params[clss][cluster]['mu'] = centroid

            self.classes_kmeans[clss] = kminstance

        return self
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions for x.

        Args:
            x (np.ndarray): An arrangement of samples. Each row of this array should be a sample.

        Returns:
            np.ndarray: A predictions vector.
        """
        assert len(x.shape) == 2
        assert x.shape[0] > 0
        assert x.shape[1] > 0

        def predict_row(xrow: np.ndarray) -> Union[int, type(None)]:
            """Makes prediction for a single row

            Returns:
                Union[int, type(None)]: Classification of the sample.
            """
            min_distance = float('inf')
            min_distance_clss = None
            for clss in self.classes_kmeans.keys():
                kminstance = self.classes_kmeans[clss]
                for cluster in self.clusters_params[clss].keys():
                    mu = self.clusters_params[clss][cluster]['mu']
                    cov = self.clusters_params[clss][cluster]['cov']
                    distance = np.abs(mahalanobis(xrow, mu, np.linalg.inv(cov)))
                    if distance < min_distance and distance < self.limit:
                        min_distance = distance
                        min_distance_clss = clss
            return min_distance_clss
        
        return np.array([predict_row(x[row]) for row in range(x.shape[0])])

if __name__ == '__main__':
    # --- generate data ---
    params = [
        (0, [[0,0], [0,5], [5,0]]),
        (1, [[5,5], [10,0]]),
        (2, [[10,10], [15,0], [5,12], [15,15]]),
        (3, [[0,15]]),
    ]
    
    x = None
    y = np.array([])
    for clss, centroids in params:
        for centroid in centroids:
            cov = np.random.random((2,2))
            samples = np.random.multivariate_normal(centroid, cov, size=100)
            y = np.append(y, np.array([clss]*100))
            x = samples if isinstance(x, type(None)) else np.vstack([x, samples])
    
    for clss in range(4):
        samples = x[y==clss]
        plt.scatter(samples[:,0], samples[:,1], label=f'Class {clss}')

    plt.legend()
    plt.grid()
    plt.show()

    classifier = MahalanobianClassifier(limit=2.5).fit(x, y)
    y_predict = classifier.predict(x)
    for clss in [0, 1, 2, 3, None]:
        samples = x[y_predict==clss]
        plt.scatter(samples[:,0], samples[:,1], label=f'Class {clss}')

    plt.legend()
    plt.grid()
    plt.show()