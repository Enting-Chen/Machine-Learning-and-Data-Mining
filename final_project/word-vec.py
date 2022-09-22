import pickle
import numpy as np
import torch
from gensim.test.utils import common_texts
from gensim.models import Word2Vec

# parameters
# 这个可以调, tf_and_id-idf.py 里面的也要调成一样的，调大调小，越大保留越少单词，这个跑完之后最后数字是剩下多少单词，记一下
min_count = 100

with open("X_splitted.npy", 'rb') as f:
    X_train = np.load(f, allow_pickle=True)
    X_test = np.load(f, allow_pickle=True)
    X_valid = np.load(f, allow_pickle=True)

print(X_train.shape)

with open("tf-idf.bin", 'rb') as f:
    X_train_tf_idf = pickle.load(f)
    X_test_tf_idf = pickle.load(f)
    X_valid_tf_idf = pickle.load(f)
    word_index_dict = pickle.load(f)

print("finished loading")

# Word2Vec() 的各个参数可以调, 这里有一些是用了默认的, 不全，vector size调，随便调，hidden_dim1上限跟着这里，也可以调其他参数
#class Word2Vec(sentences=None, corpus_file=None, vector_size=100, min_count=5, max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001, sg=0, hs=0, negative=5, ns_exponent=0.75, cbow_mean=1, hashfxn=hash, null_word=0, trim_rule=None, sorted_vocab=1, batch_words=MAX_WORDS_IN_BATCH, compute_loss=False, callbacks=(), comment=None, max_final_vocab=None, shrink_windows=True)
# model = Word2Vec(X_train, min_count=min_count,alpha=0.025, window=50, 
#                  vector_size=300, workers=5, epochs=15, seed=1)
model = Word2Vec(X_train, min_count=min_count,alpha=0.05, window=50, 
                 vector_size=1000, workers=1, epochs=10, seed=1)
print(len(model.wv))
print("calulated word2vec")

X_train_word2vec = np.array([
    np.mean([model.wv[word]
             for word in X_train[i] if word in word_index_dict], axis=0)
    for i in range(len(X_train))
])
print("finished X_train")

X_test_word2vec = np.array([
    np.mean([model.wv[word]
             for word in X_test[i] if word in word_index_dict], axis=0)
    for i in range(len(X_test))
])

X_valid_word2vec = np.array([
    np.mean([model.wv[word]
             for word in X_valid[i] if word in word_index_dict], axis=0)
    for i in range(len(X_valid))
])

print(X_train_word2vec.shape, X_test_word2vec.shape, X_valid_word2vec.shape)
# print(X_train_word2vec[0])

with open('word2vec.bin', 'wb') as f:
    pickle.dump(X_train_word2vec, f)
    pickle.dump(X_test_word2vec, f)
    pickle.dump(X_valid_word2vec, f)
