
import numpy as np
import pandas as pd
import nltk
import pickle
import time
import string

# constants
data_file = 'Test2.csv'
binary_size = 32
maxShingleID = 2 ** binary_size - 1
nextPrime = 4294967311
hashfunc_num = 50

def read_data():
    df = pd.read_csv(data_file)
    df = df[['Id', 'Title', 'FullDescription', 'Category']]
    return df

def rand_coeffs():
    return np.random.randint(0, maxShingleID, hashfunc_num, dtype=np.uint32)

def doc_to_shingles(doc):
    t0 = time.time()

    docAsShingleSets = []
    docAsHasedShingleSets = []
    doc = doc.lower().translate(None, string.punctuation)

    shingle_list = [doc[i:i + 5] for i in range(len(doc) - 5 + 1)]
    # docAsShingleSets = set(shingle_list)

    hashed_shingle_list = [hash(single_shingle) for single_shingle in shingle_list]
    docAsHasedShingleSets = set(hashed_shingle_list)

    t = time.time() - t0
    # print t
    return docAsHasedShingleSets

def docs_to_shingles(docs):
    return np.array([doc_to_shingles(doc) for doc in docs])

def shingles_to_binary_mat(docAsHasedShingleSets):
    t0 = time.time()

    bin_mat = np.zeros([len(docAsHasedShingleSets), binary_size], dtype=int)
    for k, v in enumerate(docAsHasedShingleSets):
        to_bin = (bin(v)[2:]).zfill(32)
        bin_mat[k, :] = np.fromstring(to_bin, dtype='u1') - ord('0')

    t = time.time() - t0
    # print t

    return np.sum(bin_mat == 0, axis=0) < np.sum(bin_mat == 1, axis=0).astype(int)

def jaccard(a, b):
    seta = set(a)
    setb = set(b)
    n = len(seta.intersection(setb))
    return n / float(len(seta) + len(setb) - n)

def jaccard_num(a, b):
    a_list = np.array([int(i) for i in str(a)])
    b_list = np.array([int(i) for i in str(b)])
    return np.sum(a_list==b_list / float(hashfunc_num))

def signatues_to_buckets(signatures):
    buckets = {}

    # buckets = dict(enumerate(signatures))

    for k,sig in enumerate(signatures):
        key = ''.join(map(str, sig))
        for bucket_key in buckets.keys():
            if jaccard(bucket_key, key) < 3:
                buckets[bucket_key].append(key)
        buckets[key] = []

    return buckets

if __name__ == "__main__":
    df = read_data()
    docs = df.FullDescription

    sdocs = docs_to_shingles(docs)

    signatures = [(shingles_to_binary_mat(sdocs[i])).astype(int) for i in range(0,sdocs.shape[0])]
    # print signatures

    buckets = signatues_to_buckets(signatures)
    print buckets

    pickle.dump(buckets, open('train.pkl', 'wb'))

    # A, B = rand_coeffs(), rand_coeffs()

    # print df.FullDescription

