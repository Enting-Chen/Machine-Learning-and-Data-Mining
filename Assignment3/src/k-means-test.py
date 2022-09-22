import numpy as np
import pandas as pd

# choosing initialization
# initialization = 'random', 'distance-based', 'random-from-each-label', 'data-classes-mean'
intialization = 'data-classes-mean'

def cluster_images(test, mean):
    clusterDict = {}
    for i in range(10):
        clusterDict[i] = {}
    for y in test:
        x = y[1:]
        temp = []
        for i in enumerate(mean):
            norm = np.linalg.norm(x-mean[i[0]])
            temp.append((i[0], norm))
        best = min(temp, key=first_index)[0]
        try:
            clusterDict[best][y[0]] += 1
        except KeyError:
            clusterDict[best][y[0]] = 1
    return clusterDict

def first_index(x):
	return x[1]

# load data
test = np.load('mnist_test.npy')
print('test shape is', test.shape)

# load centroids
centroids = np.load('centroids_' + intialization + '.npy')
print('centroids shape is', centroids.shape)

clusters = cluster_images(test, centroids)
for i in range(10):
    print("Cluster " + str(i) + " has " + str(len(clusters[i])) + " different digits:" ) 
    print(clusters[i])
    
cluster_df = pd.DataFrame.from_dict(clusters)
cluster_df = cluster_df.fillna(0)
cluster_df = cluster_df.sort_index()
print(cluster_df)
max_df = cluster_df.max().to_frame()
#print(max_df)
acc=max_df.sum()/10000
print('The accuracy is: ')
print(acc)
cluster_df.to_excel("output_randfromeachlabel.xlsx") 
