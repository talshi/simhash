
import numpy as np
import pandas as pd
import pickle
import time
import string

data_file = 'Train.csv'
binary_size = 32
maxShingleID = 2 ** binary_size - 1
bigPrime = 4294967311
hashfunc_num = 50

def read_data():
    t0 = time.time()
    print 'read data...'

    df = pd.read_csv(data_file)
    # df = df[['Id', 'Title', 'FullDescription', 'Category']]
    df = df[['FullDescription']][:1000]

    t = time.time() - t0
    print 'time elapsed:', t

    return df

def rand_coeffs():
    return np.random.randint(0, maxShingleID, hashfunc_num, dtype=np.uint32)

def doc_to_shingles(doc):

    docAsShingleSets = []
    docAsHasedShingleSets = []
    doc = doc.lower().translate(None, string.punctuation)

    shingle_list = [doc[i:i + 5] for i in range(len(doc) - 5 + 1)]

    hashed_shingle_list = [hash(single_shingle) for single_shingle in shingle_list]
    docAsHasedShingleSets = set(hashed_shingle_list)

    return docAsHasedShingleSets

def createSignature(sdoc, A, B):
    sdoc = np.array(list(sdoc))
    return np.min((A[:,np.newaxis]*sdoc+1*B[:,np.newaxis])%bigPrime, axis=1)

def shingle_to_binary_mat(shingles):
    bin_mat = np.zeros([len(shingles), binary_size], dtype=int)
    to_bin = [bin(shingle)[2:].zfill(32) for shingle in shingles]
    bin_mat = np.array([np.fromstring(v, dtype='u1') - ord('0') for v in to_bin])
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

def compare_docs(a, b):
    a_str = np.uint32(np.array(list(str(a))))
    b_str = np.uint32(np.array(list(str(b))))
    diff_c = 0
    for k in range(0, binary_size-1):
        aBitK = a_str[k]
        bBitK = b_str[k]
        if aBitK != bBitK:
            diff_c += 1
        if diff_c == 3:
            return False
    return True

def findBuckets(buckets, signature):
    key = ''.join(map(str, signature))

    for bucket_key in buckets.keys():
        if compare_docs(bucket_key, key):
            buckets[bucket_key].append(key)
            return buckets
    buckets[key] = []
    return buckets

if __name__ == "__main__":
    iter_counter = 0
    print_counter = 1000
    df = read_data()
    docs = df.FullDescription
    docs_num = docs.size
    A, B = rand_coeffs(), rand_coeffs()
    buckets = {}

    for doc in docs:
        iter_counter += 1
        t0 = time.time()
        sdoc = doc_to_shingles(doc)
        signature = createSignature(sdoc, A, B)
        lsh_signature = shingle_to_binary_mat(signature).astype(int)

        buckets = findBuckets(buckets, lsh_signature)
        t = time.time() - t0
        if iter_counter % print_counter == 0:
            print 'time elapsed: ', t / docs_num

    print buckets

    # export buckets to pickle file
    pickle.dump(buckets, open('train.pkl', 'wb'))
