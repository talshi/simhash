
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
bigPrime = 4294967311
hashfunc_num = 50

def read_data():
    t0 = time.time()
    print 'read data...'

    df = pd.read_csv(data_file)
    df = df[['Id', 'Title', 'FullDescription', 'Category']]

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
    # docAsShingleSets = set(shingle_list)

    hashed_shingle_list = [hash(single_shingle) for single_shingle in shingle_list]
    docAsHasedShingleSets = set(hashed_shingle_list)


    return docAsHasedShingleSets

def docs_to_shingles(docs):
    t0 = time.time()
    print 'shingle all documents...'
    shingles = np.array([doc_to_shingles(doc) for doc in docs])
    t = time.time() - t0
    print 'time elapsed:', t
    return shingles

def createSignature(sdoc, A, B):
    return np.min((A[:,np.newaxis]*sdoc+1*B[:,np.newaxis])%bigPrime, axis=1)

def createSignatures(sdocs):
    t0 = time.time()
    print 'create signatures for all documents...'

    sdocs_mat = np.array([list(sdoc) for sdoc in sdocs])
    A, B = rand_coeffs(), rand_coeffs()
    sigs = np.array([createSignature(sdoc, A, B) for sdoc in sdocs_mat])

    t = time.time() - t0
    print 'time elapsed:', t
    return sigs

def shingles_to_binary_mat(docAsHasedShingleSets):
    bin_mat = np.zeros([len(docAsHasedShingleSets), binary_size], dtype=int)
    for k, v in enumerate(docAsHasedShingleSets):
        to_bin = (bin(v)[2:]).zfill(32)
        bin_mat[k, :] = np.fromstring(to_bin, dtype='u1') - ord('0')

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

def findBuckets(signatures):
    t0 = time.time()
    print 'find buckets...'
    buckets = {}

    for k,sig in enumerate(signatures):
        key = ''.join(map(str, sig))
        for bucket_key in buckets.keys():
            if jaccard(bucket_key, key) < 3:
                buckets[bucket_key].append(key)
        buckets[key] = []

    t = time.time() - t0
    print 'time elapsed:', t

    return buckets

if __name__ == "__main__":
    t0 = time.time()
    df = read_data()
    docs = df.FullDescription
    docs_num = docs.size

    sdocs = docs_to_shingles(docs)
    # print sdocs

    signatures = createSignatures(sdocs)

    print 'creating binary matrix...'
    lsh_signatures = [(shingles_to_binary_mat(sdocs[i])).astype(int) for i in range(0,sdocs.shape[0])]
    # print signatures

    buckets = findBuckets(lsh_signatures)

    t = time.time() - t0

    print buckets
    print 'total time elapsed: ', t/docs_num

    # export buckets to pickle file
    pickle.dump(buckets, open('train.pkl', 'wb'))
