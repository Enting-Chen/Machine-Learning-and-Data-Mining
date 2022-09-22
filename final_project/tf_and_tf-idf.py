# Importing required modules
import pickle
import numpy as np
from nltk.tokenize import word_tokenize

min_count = 100
#这里要跟着调

with open("X_splitted.npy", 'rb') as f:
    X_train = np.load(f, allow_pickle=True)
    X_test = np.load(f, allow_pickle=True)
    X_valid = np.load(f, allow_pickle=True)

print("original file")
print(X_train[0][0:10])
print(X_test[0][0:10])
print(X_valid[0][0:10])

# get word index dictionary, collection frequency dictionary from X_train


def get_dicts(X_train):
    word_set = set()
    collection_freq_dict = {}
    for review in X_train:
        for word in review:
            if word not in word_set:
                word_set.add(word)
                collection_freq_dict[word] = 1
            else:
                collection_freq_dict[word] += 1
    return word_set, collection_freq_dict


print("TF and TF-IDF")
word_set, collection_freq_dict = get_dicts(X_train)
print("got word index dict,", len(word_set), "total")
print(list(word_set)[0:10])


def remove_low_freq_words(X_train, threshold, word_set):
    word_set_copy = word_set.copy()
    for word in word_set_copy:
        if collection_freq_dict[word] < threshold:
            word_set.remove(word)
    return word_set


print("remove low frequency words")
word_set = remove_low_freq_words(X_train, min_count, word_set)
print("removed low frequency words,", len(word_set), "words left")
print(list(word_set)[0:10])

# Term Frequency (TF)


def get_tf_dict(review):
    tf_dict = {}
    for word in review:
        if word in tf_dict:
            tf_dict[word] += 1
        else:
            tf_dict[word] = 1
    return tf_dict


def tf(word, review, tf_dict):
    return tf_dict[word] / len(review)


def get_df_dict(X):
    df_dict = {}
    for review in X:
        for word in review:
            if word in df_dict:
                df_dict[word] += 1
            else:
                df_dict[word] = 1
    return df_dict

# Inverse Document Frequency (IDF)


def idf(word, N, df_dict):
    return np.log10(N / df_dict[word])

# Commented out this part, as very few words are removed from the vocabulary
# remove low df words
# def remove_low_df_words(X, threshold, word_set, df_dict):
#     word_set_copy = word_set.copy()
#     for word in word_set_copy:
#         if df_dict[word] > threshold*len(X):
#             word_set.remove(word)
#     return word_set

# word_set = remove_low_df_words(X_train, 0.5, word_set, df_dict)

# print("removed low idf words ", len(word_set), " words left")


def get_word_index_dict(word_set):
    word_set = list(word_set)
    word_set.sort()
    word_index_dict = {}
    for i, word in enumerate(word_set):
        word_index_dict[word] = i
    return word_index_dict


word_index_dict = get_word_index_dict(word_set)
print("got word dict")
# print(word_index_dict.keys())

df_dict_train = get_df_dict(X_train)
df_dict_test = get_df_dict(X_test)
df_dict_valid = get_df_dict(X_valid)


def tf_vec(review, N, df_dict):
    tf_vector = np.zeros(len(word_index_dict))
    tf_dict = get_tf_dict(review)
    for word in review:
        if word in word_index_dict:
            tf_vector[word_index_dict[word]] = tf(word, review, tf_dict)
    return tf_vector


def tf_idf_vec(review, N, df_dict):
    tf_idf_vector = np.zeros(len(word_index_dict))
    tf_dict = get_tf_dict(review)
    for word in review:
        if word in word_index_dict:
            tf_idf_vector[word_index_dict[word]] = tf(
                word, review, tf_dict) * idf(word, N, df_dict)
    return tf_idf_vector


X_train_tf_idf = np.array(
    [tf_idf_vec(review, len(X_train), df_dict_train) for review in X_train])
X_test_tf_idf = np.array(
    [tf_idf_vec(review, len(X_test), df_dict_test) for review in X_test])
X_valid_tf_idf = np.array(
    [tf_idf_vec(review, len(X_valid), df_dict_valid) for review in X_valid])

X_train_tf = np.array(
    [tf_vec(review, len(X_train), df_dict_train) for review in X_train])
X_test_tf = np.array(
    [tf_vec(review, len(X_test), df_dict_test) for review in X_test])
X_valid_tf = np.array(
    [tf_vec(review, len(X_valid), df_dict_valid) for review in X_valid])

with open("tf.bin", "wb") as f:
    pickle.dump(X_train_tf, f)
    pickle.dump(X_test_tf, f)
    pickle.dump(X_valid_tf, f)
    pickle.dump(word_index_dict, f)

with open("tf-idf.bin", 'wb') as f:
    pickle.dump(X_train_tf_idf, f)
    pickle.dump(X_test_tf_idf, f)
    pickle.dump(X_valid_tf_idf, f)
    pickle.dump(word_index_dict, f)

print(X_train_tf.shape, X_test_tf.shape, X_valid_tf.shape)
print(X_train_tf[0:10])
print(X_test_tf[0:10])
print(X_valid_tf[0:10])

print(X_train_tf_idf.shape, X_test_tf_idf.shape, X_valid_tf_idf.shape)

print("finished")
