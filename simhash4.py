
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
# docs_num = 2000

number_new_buckets = 0
number_added_to_buckets = 0
train_number_of_new_bucket = 0

exact = 0
one_bit_excat = 0
two_bit_excat = 0

train = "TRAIN"
test = "TEST"
zero_bit = "ZERO_BIT"
one_bit = "ONE_BIT"
two_bit = "TWO_BIT"


############################################################

def read_data():
    t0 = time.time()
    print 'read data...'

    df = pd.read_csv(data_file)
    df = df[['FullDescription']][:]

    t = time.time() - t0
    print 'time elapsed:', t

    return df

############################################################

def rand_coeffs():
    return np.random.randint(0, maxShingleID, hashfunc_num, dtype=np.uint32)

############################################################

def doc_to_shingles(doc):

    doc = doc.lower().translate(None, string.punctuation)

    doc_split = doc.split(" ")

    shingle_list = [" ".join(doc_split[i:i + 5]) for i in range(len(doc_split) - 5 + 1)]

    hashed_shingle_list = [(hash(single_shingle) & 0xffffffff) for single_shingle in shingle_list]
    docAsHasedShingleSets = set(hashed_shingle_list)

    return docAsHasedShingleSets

############################################################

def createSignature(sdoc, A, B):
    sdoc = np.array(list(sdoc))
    x = np.min((A[:,np.newaxis]*sdoc+1*B[:,np.newaxis])%bigPrime, axis=1)
    # print x
    return x

############################################################

def shingle_to_binary_mat(shingles):
    bin_mat = np.zeros([len(shingles), binary_size], dtype=int)
    to_bin = [bin(shingle)[2:].zfill(32) for shingle in shingles]
    bin_mat = np.array([np.fromstring(v, dtype='u1') - ord('0') for v in to_bin])
    return (np.sum(bin_mat == 0, axis=0) < np.sum(bin_mat == 1, axis=0)).astype(int)

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

def jaccard(a, b):
    seta = doc_to_shingles(a)
    setb = set(b)
    n = len(seta.intersection(setb))
    return n / (float(len(seta) + len(setb) - n))

########################################################################################################
def printDocs(sign1, sign2,status):
        print status
        doc = df.loc[dicIDsignature[sign1], 'FullDescription']
        print "doc BUCKET :\n",doc
        doc = df.loc[dicIDsignature[sign2], 'FullDescription']
        print "doc SIGN:\n",doc
########################################################################################################
def check_in_buckets(buckets, key, lineNumber,error_num, bit):

        if bit == "ZERO_BIT":
            if key in buckets:
                buckets[key].append(lineNumber)
                # print("exact hit")
                global  exact
                exact += 1
                return True, buckets , key
            return False, buckets, None

        if bit == "ONE_BIT":
            for i in range(32):
                # split string to array
                temp = list(key)
                if temp[i] == '0':
                    temp[i] = '1'
                else:
                    temp[i] = '0'
                bit_changed_signature = "".join(temp)
                if bit_changed_signature in buckets:
                    # print('--found 1 bit different signature!!')
                    buckets[bit_changed_signature].append(lineNumber)
                    global one_bit_excat
                    one_bit_excat += 1
                    return True, buckets ,bit_changed_signature
            return False, buckets, None

        if bit == "TWO_BIT":
            for i in range(32):
                # split string to array
                temp = list(key)
                # replace bit with complimentary
                if temp[i] == '0':
                    temp[i] = '1'
                else:
                    temp[i] = '0'
                for j in range(32):
                    # make sure no duplications are used
                    if j > i:
                        temp2 = temp[:]
                        if temp2[j] == '0':
                            temp2[j] = '1'
                        else:
                            temp2[j] = '0'
                        bit_changed_signature = "".join(temp2)
                        # print(bit_changed_signature, '  i: ', i, '  j: ', j)
                        if bit_changed_signature in buckets:
                            buckets[bit_changed_signature].append(lineNumber)
                            # print('--found 2 bit different signature!!')
                            global two_bit_excat
                            two_bit_excat += 1
                            return True, buckets, bit_changed_signature
            return False, buckets, None

#########################################################################################################
def findBuckets(buckets, signature, lineNumber, status, error_num, docs_dict, i, doc):
    # print signature

    key = ''.join(map(str, signature))

    if len(dicIDsignature) == 0:
        addToDic(key, lineNumber)
        buckets.setdefault(key,[]).append(lineNumber)
        return buckets, error_num

    addToDic(key, lineNumber)

    jac = 0

    result, buckets ,bucket_key = check_in_buckets(buckets, key, lineNumber,error_num, zero_bit)
    if result == True:
        jac = jaccard(doc, docs_dict[i-1])
        if jac >= 0.8:
            error_num += 1
        return buckets, error_num

    result, buckets, bucket_key = check_in_buckets(buckets, key, lineNumber,error_num, one_bit)
    if result == True:
        jac = jaccard(doc, docs_dict[i-1])
        if jac >= 0.8:
            error_num += 1
        return buckets, error_num

    result, buckets, bucket_key = check_in_buckets(buckets, key, lineNumber,error_num, two_bit)
    if result == True:
        jac = jaccard(doc, docs_dict[i-1])
        if jac >= 0.8:
            error_num += 1
        return buckets, error_num

    # Not found 0,1,2 bits - need to onen new bucket with key sign
    if jac >= 0.8:
        error_num += 1

    buckets[key] = []
    buckets[key].append(lineNumber)
    global train_number_of_new_bucket
    train_number_of_new_bucket += 1

    return buckets ,error_num

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
    buckets = {}
    lineNumber = 0
    error_num = 0
    train_accuracy = 1
    i = 0
    t_train_total = 0
    docs_dict = []

    for i, doc in enumerate(docs):
        iter_counter += 1
        t0 = time.time()
        shingles_hash_set = doc_to_shingles(doc)   # return set hash shingles
        docs_dict.append(shingles_hash_set)
        signature = shingle_to_binary_mat(shingles_hash_set).astype(int)

        buckets, error_num = findBuckets(buckets, signature, lineNumber, train, error_num, docs_dict, i, doc)
        lineNumber += 1
        error = error_num / iter_counter
        train_accuracy = 1 - error

        t = time.time() - t0
        t_train_total += t
        if iter_counter % print_counter == 0:
            print "ITER #", i
            print 'time elapsed: ', t_train_total / docs_num
            print 'train accuracy:', train_accuracy

    # export buckets to pickle file
    pickle.dump(buckets, open('train.pkl', 'wb'))

    ###################### TEST AREA ###################

    iter_counter = 0
    error_num = 0
    test_accuracy = 1
    df_test = read_dataTest()
    docs_test = df_test.FullDescription
    docs_num_test = docs_test.size
    t_test_total = 0

    for i, doc in enumerate(docs_test):
        iter_counter += 1
        t0 = time.time()
        shingles_hash_set = doc_to_shingles(doc)  # return set hash shingles
        docs_dict.append(shingles_hash_set)
        signature = shingle_to_binary_mat(shingles_hash_set).astype(int)

        buckets, error_num = findBuckets(buckets, signature, lineNumber, train, error_num, docs_dict, i, doc)
        lineNumber += 1
        error = error_num / iter_counter
        accuracy = 1 - error

        t = time.time() - t0
        t_test_total += t
        if iter_counter % print_counter == 0:
            print "ITER #", i
            print 'time elapsed: ', t_test_total / docs_num
            print 'test accuracy:', test_accuracy


    print "::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n\n\n"
    print 'train accuracy:', train_accuracy
    print 'train iteration avg time: ', t_train_total / docs_num
    print 'test accuracy:', test_accuracy
    print 'test iteration avg time: ', t_test_total / docs_num
    # print 'excat: ', exact
    # print  'one_bit_excat:', one_bit_excat
    # print 'two_bit_excat:', two_bit_excat
    # print 'sum one,two :', one_bit_excat + two_bit_excat
    # print 'ERROR: ', error_num
    print 'number_of_new_bucket (train) :', train_number_of_new_bucket
