import numpy as np
import gmm
import classifier
import pandas as pd

# Load data
train = np.load('mnist_train.npy')
test = np.load('mnist_test.npy')
train_data = train[:, 1:].astype(np.float64)/255
test_data = test[:, 1:].astype(np.float64)/255
train_labels = train[:, 0]
test_labels = test[:, 0]

label_set = set(train_labels)
precisions = []

k = 10

print('learning {} components'.format(k))
model = classifier.classifier(k, covariance_type='full',
                              model_type='gmm',
                              means_init_heuristic='distance-based',
                              verbose=True)
model.fit(train_data, train_labels)

clusters = model.predict(test_data, test_labels)
print(clusters)
matching_dict = {'test_labels': test_labels, 'predicted_labels': clusters}
cluster_df = pd.DataFrame.from_dict(clusters)
cluster_df = cluster_df.fillna(0)
cluster_df = cluster_df.sort_index()
print(cluster_df)
max_df = cluster_df.max().to_frame()
#print(max_df)
acc=max_df.sum()/10000
print('The accuracy is: ')
print(acc)
cluster_df.to_excel("confusion_matrix_full_mean.xlsx") 

#for i in range(10):
#    print("Cluster " + str(i) + " has " + str(len(clusters[i])) + " different digits:" ) 
#    print(clusters[i])