import numpy as np
import gmm

class classifier:

    def __init__(self, n_components,
                 means_init_heuristic='random',
                 covariance_type='diag',
                 model_type='gmm', means=None, verbose=False):

        self.n_components = n_components
        self.means_init_heuristic = means_init_heuristic
        self.covariance_type = covariance_type
        self.model_class = gmm.gmm
        self.means = means
        self.verbose = verbose

        # self.models = dict()

    def fit(self, x, labels):

        label_set = set(labels)

        # for label in label_set:

        # x_subset = x[np.in1d(labels, label)]
        self.model  = self.model_class(
            self.n_components, covariance_type=self.covariance_type,
            verbose=self.verbose)
        self.model.fit(
            x, means_init_heuristic=self.means_init_heuristic,
            labels=labels)

    def predict(self, x, labels):
        print(x.shape, labels.shape)
        clusterDict = {}
        for i in range(10):
            clusterDict[i] = {}
        for i in range(x.shape[0]):
            xi = x[i]
            label = labels[i]
            best = self.model.predict(xi)
            try:
                clusterDict[best][label] += 1
            except KeyError:
                clusterDict[best][label] = 1
        return clusterDict
        # n = x.shape[0]
        # print('predicting {} samples'.format(n))

        # likelihoods = np.ndarray(shape=(len(label_set), n))

        # for label in label_set:
        #     likelihoods[label] = self.model.predict(x)

        # predicted_labels = np.argmax(likelihoods, axis=0)

        # return predicted_labels
