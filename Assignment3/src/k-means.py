import numpy as np
import matplotlib.pyplot as plt 

# choosing initialization
# initialization = 'random', 'distance-based', 'random-from-each-label', 'data-classes-mean'
initialization = 'data-classes-mean'

def get_cluster_centroids(images, k):
	print('k is', k)
	oldMean = np.zeros((k, 784))
	trueMean = initial_centroids
	iteration = 1
	while not same_mean(trueMean, oldMean):
		oldMean = trueMean
		clusters = cluster_images(images, trueMean)
		trueMean = find_new_mean(clusters)
		print("iteration", iteration)
		iteration += 1
	return trueMean

def cluster_images(images, mean):
	clusterDict = {}
	for x in images:
		temp = []
		for i in enumerate(mean):
			norm = np.linalg.norm(x-mean[i[0]])
			temp.append((i[0], norm))
		best = min(temp, key=first_index)[0]
		try:
			clusterDict[best].append(x)
		except KeyError:
			clusterDict[best] = [x]
	return clusterDict

def same_mean(trueMean, oldMean):
	oldList, newList = [tuple(i) for i in oldMean], [tuple(i) for i in trueMean]
	oldMeanSet, trueMeanSet = set(oldList), set(newList)
	return oldMeanSet == trueMeanSet	

def find_new_mean(clusters):
    clusterKeys = sorted(clusters.keys())
    newMean = []
    for k in clusterKeys:
        newMean.append(np.mean(clusters[k], axis=0, dtype=None, out=None, keepdims=False))
    return newMean

def first_index(x):
	return x[1]

def show_clusters():
	mean = get_cluster_centroids(images, k)
	# for mean in retList:
	for i,m in enumerate(mean):
		plt.subplot(5,4,i+1)
		plt.imshow(m.reshape(28, 28), cmap='gray')
		plt.axis('off')
	plt.show()
	return mean

# load data
train = np.load('mnist_train.npy')
print('train shape is', train.shape)

# initial centroids, randomly pick 10 images from training set, with labels 0-9
# intial centroids is a zero matrix with shape (10, 784)
# each row is a 784-dim vector, which is a pixel value of an image
# each column is a pixel value of an image

labels = train[:,0]
images = train[:,1:]

k = 10
if initialization == 'random':
	initial_centroids = images[np.random.choice(images.shape[0], k, replace=False),:]
elif initialization == 'distance-based':
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
elif initialization == 'random-from-each-label':
	initial_centroids = np.zeros((k, 785))
	for i in range(k):
		# for each label, randomly pick one image
		temp = train[train[:,0]==i]
		initial_centroids[i] = temp[np.random.randint(0, temp.shape[0])]
	initial_centroids = initial_centroids[:,1:]
elif initialization == 'data-classes-mean':
	x = images
	n, d = x.shape
	print(n, labels.shape)

	assert labels.shape[0] == n, 'labels and data shapes must match'

	label_set = set(labels)
	n_labels = len(label_set)

	means = np.ndarray(shape=(n_labels, d))

	for l in label_set:
		matches = np.in1d(labels, l)
		means[l] = x[matches].mean(0)

	initial_centroids = means

print('initial centroids shape is', initial_centroids.shape)

centroids = show_clusters()
np.save('centroids_' + initialization + '.npy', centroids)