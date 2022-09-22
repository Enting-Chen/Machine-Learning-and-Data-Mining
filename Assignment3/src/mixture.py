from datetime import datetime

import numpy as np

EPS = np.finfo(float).eps

class mixture:

    def __init__(self, n_components, init_params='wm', n_iter=100, tol=1e-3,
                 covariance_type='diag', min_covar=1e-4, verbose=False):

        #: number of components in the mixture
        self.n_components = n_components
        #: params to init
        self.init_params = init_params
        #: max number of iterations
        self.n_iter = n_iter
        #: convergence threshold
        self.tol = tol
        self.covariance_type = covariance_type
        self.min_covar = min_covar
        self.verbose = verbose

        k = self.n_components

        self.weights = np.array([1 / k for _ in range(k)])
        self.means = None
        self.covars = None

        self.converged_ = False

    def fit(self, x, means_init_heuristic='random', means=None, labels=None):

        k = self.n_components
        n = x.shape[0]
        d = x.shape[1]

        self.means = np.ndarray(shape=(k, d))

        # initialization of the means
        if 'm' in self.init_params:
            if self.verbose:
                print('using {} heuristic to initialize the means'
                      .format(means_init_heuristic))
            if means_init_heuristic == 'random':
                self.means = x[np.random.choice(x.shape[0], k, replace=False),:]
            elif means_init_heuristic == 'data_classes_mean':
                if labels is None:
                    raise ValueError(
                        'labels required for data_classes_mean init')
                self.means = _data_classes_mean_init(x, labels)    
            elif means_init_heuristic == 'distance-based':
                images = x
                initial_centroids = np.zeros((k, 784))
                initial_centroids[0] = images[np.random.randint(0, images.shape[0])]
                for i in range(1, k):
                    print('i is', i)
                    # for each remaining images, find the distance to the closest centroid
                    # and pick the furthest one
                    temp = np.zeros((images.shape[0], 2))
                    for j in range(images.shape[0]):
                        temp[j,0] = np.linalg.norm(images[j]-initial_centroids[0])
                        temp[j,1] = j
                        for l in range(1, i):
                            # find closest centroid
                            norm = np.linalg.norm(images[j]-initial_centroids[l])
                            if norm < temp[j, 0]:
                                temp[j,0] = norm
                    temp = temp[temp[:,0].argsort()]
                    initial_centroids[i] = images[int(temp[-1,1])]
                self.means = initial_centroids
            elif means_init_heuristic == 'random-from-each-label':
                initial_centroids = np.zeros((k, 784))
                for i in range(k):
                    # for each label, randomly pick one image
                    temp = labels[labels==i]
                    initial_centroids[i] = temp[np.random.randint(0, temp.shape[0])]
                self.means = initial_centroids

        # initialization of the covars
        if 'c' in self.init_params:
            if self.verbose:
                print('initializing covars')
            cv = np.cov(x.T) + self.min_covar * np.eye(x.shape[1])
            if self.covariance_type == 'diag':
                self.covars = np.tile(np.diag(cv), (k, 1))
            elif self.covariance_type == 'full':
                self.covars = np.tile(cv, (k, 1, 1))

        start = datetime.now()

        iterations = 0

        prev_log_likelihood = None
        current_log_likelihood = -np.inf

        while iterations <= self.n_iter:

            elapsed = datetime.now() - start

            prev_log_likelihood = current_log_likelihood

            # expectation step
            log_likelihoods, responsibilities = self.score_samples(x)
            current_log_likelihood = log_likelihoods.mean()

            if self.verbose:

                print('[{:02d}] likelihood = {} (elapsed {})'
                      .format(iterations, current_log_likelihood, elapsed))

            if prev_log_likelihood is not None:
                change = abs(current_log_likelihood - prev_log_likelihood)
                if change < self.tol:
                    self.converged_ = True
                    break

            self._do_mstep(x, responsibilities)

            iterations += 1

        end = datetime.now()

        elapsed = end - start

        print('converged in {} iterations in {}'
              .format(iterations, elapsed))

    def _do_mstep(self, x, z):

        weights = z.sum(axis=0)
        weighted_x_sum = np.dot(z.T, x)
        inverse_weights = 1.0 / (weights[:, np.newaxis] + 10 * EPS)

        self.weights = (weights / (weights.sum() + 10 * EPS) + EPS)
        self.means = weighted_x_sum * inverse_weights


    def score_samples(self, x):

        log_support = self._log_support(x)

        lpr = log_support + np.log(self.weights)
        logprob = np.logaddexp.reduce(lpr, axis=1)
        responsibilities = np.exp(lpr - logprob[:, np.newaxis])

        return logprob, responsibilities

    def predict(self, x):
        
        x = x.reshape(1, 784)
        # print(np.argmax(self._log_support(x)))

        return np.argmax(self._log_support(x))
        # return np.sum(np.exp(self._log_support(x)), 1)

def _data_classes_mean_init(x, labels):

    n, d = x.shape

    assert labels.shape[0] == n, 'labels and data shapes must match'

    label_set = set(labels)
    n_labels = len(label_set)

    means = np.ndarray(shape=(n_labels, d))

    for l in label_set:
        matches = np.in1d(labels, l)
        means[l] = x[matches].mean(0)

    return means
