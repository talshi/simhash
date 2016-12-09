
from __future__ import division
import numpy as np
import pandas as pd
import pickle
import time
import string

data_file = 'Train.csv'
data_file_test = 'Test.csv'
binary_size = 32
maxShingleID = 2 ** binary_size - 1
bigPrime = 4294967311
hashfunc_num = 50

number_new_buckets = 0
number_added_to_buckets = 0

train = "TRAIN"
test = "TEST"
############################################################
# def jaccard(bucket_sign,new_sign):
#
#     a_str = np.uint32(np.array(list(str(bucket_sign))))
#     b_str = np.uint32(np.array(list(str(new_sign))))
#     up = 0
#     down = 0
#     for k in range(0, binary_size):
#         a = a_str[k]
#         b = b_str[k]
#         if((a != 0) | (b != 0)):
#             down += 1
#             if(a == b):
#                 up += 1
#
#     if(down == 0):
#         return
#     resJaccard = (up / down)
#     print "jaccard" ,resJaccard
#     return resJaccard
############################################################
def jaccard(a, b):
    seta = doc_to_shingles(dicIDsignature[a])
    setb = doc_to_shingles(dicIDsignature[b])
    n = len(seta.intersection(setb))
    return n / float(len(seta) + len(setb) - n)
#
#     # OFFLINE  AND  ONLINE !!!!!!!
#
#     # a -> bucket
#     # b -> sign
#     # dicIDsignature  { line number , sign }  |  file -> { line numer , doc }
#     # 1)  create shingls to doc of bucket = > shinglsBucket
#     # 2)  create shingls to sign -> dicIDsign .
#     # 3) jec = check jeccard according
#     # 4) if jec > 0.8
#     # 4.1) sign need to be in this bucket
#     # 5) zao
#



############################################################
def read_data():
    t0 = time.time()
    print 'read data...'

    df = pd.read_csv(data_file)
    # df = df[['Id', 'Title', 'FullDescription', 'Category']]
    df = df[['FullDescription']][:30]

    t = time.time() - t0
    print 'time elapsed:', t

    return df
############################################################
def rand_coeffs():
    return np.random.randint(0, maxShingleID, hashfunc_num, dtype=np.uint32)
############################################################
def doc_to_shingles(doc):

    docAsShingleSets = []
    docAsHasedShingleSets = []
    doc = doc.lower().translate(None, string.punctuation)

    shingle_list = [doc[i:i + 5] for i in range(len(doc) - 5 + 1)]

    hashed_shingle_list = [hash(single_shingle) for single_shingle in shingle_list]
    docAsHasedShingleSets = set(hashed_shingle_list)

    return docAsHasedShingleSets
############################################################
def createSignature(sdoc, A, B):
    sdoc = np.array(list(sdoc))
    return np.min((A[:,np.newaxis]*sdoc+1*B[:,np.newaxis])%bigPrime, axis=1)

def shingle_to_binary_mat(shingles):
    bin_mat = np.zeros([len(shingles), binary_size], dtype=int)
    to_bin = [bin(shingle)[2:].zfill(32) for shingle in shingles]
    bin_mat = np.array([np.fromstring(v, dtype='u1') - ord('0') for v in to_bin])
    return np.sum(bin_mat == 0, axis=0) < np.sum(bin_mat == 1, axis=0).astype(int)
############################################################
def compare_docs(a, b):
    a_str = np.uint32(np.array(list(str(a))))
    b_str = np.uint32(np.array(list(str(b))))
    diff_c = 0
    for k in range(0, binary_size):
        aBitK = a_str[k]
        bBitK = b_str[k]
        if aBitK != bBitK:
            diff_c += 1
        if diff_c > 1:
            return False
    return True
############################################################
def addToDic(key, lineNumber):
    dicIDsignature.update({key: lineNumber})
############################################################
def findBuckets(buckets, signature, lineNumber, status):
    key = ''.join(map(str, signature))

    if status == "TRAIN":
        addToDic(key, lineNumber)

    for bucket_key in buckets.keys():

        jec = jaccard(bucket_key, key)  # need get 0.8  for insert to excist buecket

        if compare_docs(bucket_key, key):  # need compare
            buckets[bucket_key].append(lineNumber)
            print "------------------------------------------------------------------------------------------"
            print "*************Insert to buckets -> NEED TO BE MORE 0.8 ACCORDING JECCARD *********", jec
            print "sign1->Buecket-[",bucket_key,"]"
            print "sign2->newSign[",key,"]"
            if(status == "TEST"):
                global number_added_to_buckets
                number_added_to_buckets += 1
            return buckets
    buckets[key] = []
    buckets[key].append(lineNumber)
    if(status == "TEST"):
        global number_new_buckets
        number_new_buckets += 1
    return buckets
############################################################

if __name__ == "__main__":
    dicIDsignature = {}  # {'key(ID)': 'value(SIGNATURE)'}
    iter_counter = 0
    print_counter = 1000
    df = read_data()
    docs = df.FullDescription
    docs_num = docs.size
    A, B = rand_coeffs(), rand_coeffs()
    buckets = {}
    lineNumber = 0

    for doc in docs:
        iter_counter += 1
        t0 = time.time()
        sdoc = doc_to_shingles(doc)
        signature = createSignature(sdoc, A, B)
        lsh_signature = shingle_to_binary_mat(signature).astype(int)

        buckets = findBuckets(buckets, lsh_signature, lineNumber,train)
        lineNumber += 1

        t = time.time() - t0
        if iter_counter % print_counter == 0:
            print 'time elapsed: ', t / docs_num

   #print buckets
   #print dicIDsignature
  #  export buckets to pickle file
  # pickle.dump(buckets, open('train.pkl', 'wb'))

    ###################### TEST AREA ###################

    def read_dataTest():
        t0 = time.time()
        print 'read data Test...'

        dfTest = pd.read_csv(data_file_test)
        # df = df[['Id', 'Title', 'FullDescription', 'Category']]
        dfTest = dfTest[['FullDescription']][:30]

        t = time.time() - t0
        print 'time elapsed:', t

        return dfTest

    df_test = read_dataTest()
    docs_test = df_test.FullDescription
    docs_num_test = docs_test.size

    for doc in docs_test:
        sdoc = doc_to_shingles(doc)
        signature = createSignature(sdoc, A, B)
        lsh_signature = shingle_to_binary_mat(signature).astype(int)

        buckets = findBuckets(buckets, lsh_signature, lineNumber,test)# "TEST"
        lineNumber += 1

    print buckets
    print "NEW BUCKETS IN TEST : ", number_new_buckets
    print "ADDED TO BUCKETS IN TEST: ", number_added_to_buckets

    # x =  '00011111110000000000001000000000'
    # y =  '00001111110000000000001000000000'  # 7 / 8
    #
    # print jaccard(x,y)