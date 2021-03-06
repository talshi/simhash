
from __future__ import division
import numpy as np
import pandas as pd
import pickle
import time
import string

data_file = 'Train20.csv'
data_file_test = 'Test.csv'
binary_size = 32
maxShingleID = 2 ** binary_size - 1
bigPrime = 4294967311
hashfunc_num = 50
docs_num = 100

number_new_buckets = 0
number_added_to_buckets = 0

train = "TRAIN"
test = "TEST"

############################################################

def read_data():
    t0 = time.time()
    print 'read data...'

    df = pd.read_csv(data_file)
    df = df[['FullDescription']][:docs_num]

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

    doc_split = doc.split(" ")

    shingle_list = [" ".join(doc_split[i:i + 5]) for i in range(len(doc_split) - 5 + 1)]

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

# ############################################################
#
# def compare_docs(a, b):
#     a_str = np.uint32(np.array(list(str(a))))
#     b_str = np.uint32(np.array(list(str(b))))
#     diff_c = 0
#     for k in range(1, int(binary_size/4)):
#         aBitK = a_str[-k]
#         bBitK = b_str[-k]
#         if aBitK != bBitK:
#             diff_c += 1
#         if diff_c > 1:
#             return False
#     for k in range(int((binary_size/4)+1), binary_size+1):
#         aBitK = a_str[-k]
#         bBitK = b_str[-k]
#         if aBitK != bBitK:
#             diff_c += 1
#         if diff_c > 1:
#             return False
#     return True
#
# ############################################################
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
        if diff_c > 3:
            return False
    return True

############################################################


def addToDic(key, lineNumber):
    dicIDsignature.update({key: lineNumber})

############################################################

def jaccard(a, b):
    seta = doc_to_shingles(df.loc[dicIDsignature[a], 'FullDescription'])
    setb = doc_to_shingles(df.loc[dicIDsignature[b], 'FullDescription'])
    n = len(seta.intersection(setb))
    return n / float(len(seta) + len(setb) - n)

############################################################
def printDocs(sign1, sign2,status):
        print status
        doc = df.loc[dicIDsignature[sign1], 'FullDescription']
        print "doc BUCKET :\n",doc
        doc = df.loc[dicIDsignature[sign2], 'FullDescription']
        print "doc SIGN:\n",doc
############################################################
def findBuckets(buckets, signature, lineNumber, status, error_num):
    key = ''.join(map(str, signature))

    if len(dicIDsignature) == 0: # if buckets number is 0 - need to add the sign to dicIDsignatue - open new bucket
        print dicIDsignature
        addToDic(key, lineNumber)   #     dicIDsignature.update({key: lineNumber})
        buckets.setdefault(key,[]).append(lineNumber)    #[key].append(lineNumber)          # a.setdefault("somekey",[]).append("bob")
        print "bucket tatus", buckets.keys()
        return buckets, error_num

    if status == "TRAIN":
        addToDic(key, lineNumber)
    elif status == "TEST":
        addToDic(key, lineNumber-docs_num)


    print "bucket tatus" , buckets.keys()
    i = 0
    for bucket_key in buckets.keys():
        flag = 0
        jac = jaccard(bucket_key, key)  # need get 0.8  for insert to excist buecket
        res_compare_to = compare_docs(bucket_key, key)
        # print jac
        print "----------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> FIRST", jac
        print "----------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", res_compare_to
        print "----------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", i
        i += 1
        printDocs(bucket_key, key, status)
        if jac >= 0.8:
            flag = 1
            print "sign1 -> Bucket [",bucket_key,"]"
            print "sign2 -> newSign[",key,"]"

        if res_compare_to:  # need compare
            if jac < 0.8:
                error_num += 1
            buckets[bucket_key].append(lineNumber)
            print "************* Insert to buckets -> NEED TO BE > 0.8 ACCORDING TO JACCARD *********", jac
            if flag == 0:
                print "sign1 -> Bucket [",bucket_key,"]"
                print "sign2 -> newSign[",key,"]"
            print "------------------------------------------------------------------------------------------"

            if(status == "TEST"):
                global number_added_to_buckets
                number_added_to_buckets += 1
            return buckets, error_num
    buckets[key] = []
    buckets[key].append(lineNumber)
    if(status == "TEST"):
        global number_new_buckets
        number_new_buckets += 1

    return buckets, error_num

############################################################

def read_dataTest():
    t0 = time.time()
    print 'read data Test...'

    dfTest = pd.read_csv(data_file_test)
    dfTest = dfTest[['FullDescription']][:docs_num]

    t = time.time() - t0
    print 'time elapsed:', t

    return dfTest

############################################################

if __name__ == "__main__":
    dicIDsignature = {}
    iter_counter = 0
    print_counter = 100
    df = read_data()
    docs = df.FullDescription
    docs_num = docs.size
    A, B = rand_coeffs(), rand_coeffs()
    buckets = {}
    lineNumber = 0
    error_num = 0
    accuracy = 1

    i = 0
    for doc in docs:
        print "\n\n+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=DOC : ",i
        i += 1
        iter_counter += 1
        t0 = time.time()
        sdoc = doc_to_shingles(doc)
        signature = createSignature(sdoc, A, B)
        lsh_signature = shingle_to_binary_mat(signature).astype(int)

        buckets, error_num = findBuckets(buckets, lsh_signature, lineNumber, train, error_num)
        lineNumber += 1
        error = error_num / iter_counter
        accuracy = 1 - error

        t = time.time() - t0
        if iter_counter % print_counter == 0:
            print 'time elapsed: ', t / docs_num
            print 'accuracy:', accuracy

    print buckets
    print 'accuracy:', accuracy

    # export buckets to pickle file
    pickle.dump(buckets, open('train.pkl', 'wb'))

    ###################### TEST AREA ###################

    # iter_counter = 0
    # error_num = 0
    # accuracy = 1
    # df_test = read_dataTest()
    # docs_test = df_test.FullDescription
    # docs_num_test = docs_test.size
    #
    # for doc in docs_test:
    #     iter_counter += 1
    #     t0 = time.time()
    #     sdoc = doc_to_shingles(doc)
    #     signature = createSignature(sdoc, A, B)
    #     lsh_signature = shingle_to_binary_mat(signature).astype(int)
    #
    #     buckets, error_num = findBuckets(buckets, lsh_signature, lineNumber,test, error_num)
    #     lineNumber += 1
    #     error = error_num / iter_counter
    #     accuracy = 1 - error
    #
    #     t = time.time() - t0
    #     if iter_counter % print_counter == 0:
    #         print 'time elapsed: ', t / docs_num
    #         print 'accuracy:', accuracy
    print "::::::::::::::::::::::::::::::::::::::::::::\n\n\n"
    print buckets
    print 'accuracy:', accuracy
    print "NEW BUCKETS IN TEST : ", number_new_buckets
    print "ADDED TO BUCKETS IN TEST: ", number_added_to_buckets

    # export buckets to pickle file
    pickle.dump(buckets, open('test.pkl', 'wb'))
