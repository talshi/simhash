
import pandas as pd
import binascii
import random
import time
import os

# constants
maxShingleID = 2 ** 32 - 1
nextPrime = 4294967311
numHashes = 10

docsNames = []
title_col = []
fd_col = []
docsAsShingleSets = {};

def read_data(csv_file):
    print '>>> read data from', csv_file, '...'
    data = pd.read_csv(csv_file)
    title_col = data.Title
    fd_col = data.FullDescription

    return title_col, fd_col

def shingle_data():
    print ">>> shingling data..."
    doc_counter = 0
    for doc in fd_col:
        words = doc.split(" ")

        docID = title_col[doc_counter]
        docsNames.append(docID)
        doc_counter = doc_counter + 1

        shingle_list = [doc[i:i + 5] for i in range(len(doc) - 5 + 1)]

        hashed_shingle_list = [binascii.crc32(single_shingle) & 0xffffffff for single_shingle in shingle_list]
        shinglesInDoc = set(hashed_shingle_list)

        docsAsShingleSets[docID] = shinglesInDoc
    return docsAsShingleSets

def write_to_file(file_name, data):
    print ">>> writing to", file_name
    file = open(file_name, 'w')
    for row in data:
        file.write(str(row))
        file.write(':')
        for v in data[row]:
            file.write(repr(v))
            file.write(' ')
        file.write('\n')
    file.close()

def read_pkl(file_name):
    docsAsShingleSets = {}
    with open(file_name, 'r') as f:
        for row in f.readlines():
            title, data = row.split(':')
            docsAsShingleSets[title] = set(data.split(" "))
    return docsAsShingleSets

def rand_coeffs(numHashes):
    randList = []

    k = numHashes

    while k > 0:
        random_int = random.randint(0, maxShingleID)

        while random_int in randList:
            random_int = random.randint(0, maxShingleID)

        randList.append(random_int)
        k = k - 1

    return randList


def CreateSignature():

    t_delta = time.time()

    print ">>> create coeffs vectors..."

    A = rand_coeffs(numHashes)
    B = rand_coeffs(numHashes)

    print ">>> create signatures..."

    signatures = []

    for id in docsNames:

        shingleIDSet = docsAsShingleSets[id]
        signature = []

        for i in range(0, numHashes):

            # must be bigger than the hash
            minHashCode = nextPrime + 1

            for shingleID in shingleIDSet:

                # (A*x+B) % p
                hash = (A[i] * shingleID + B[i]) % nextPrime

                # replace minHashCode only if hash is smaller
                if hash < minHashCode:
                    minHashCode = hash

            signature.append(minHashCode)
        signatures.append(signature)

    t = time.time() - t_delta

    print 'creating signatures run time: %.2fsec' % t

def jacard_compare():
    pass

if __name__ == "__main__":
    output_file = 'train.pkl'

    # if not os.path.isfile(output_file):
    file_name = 'Test2.csv'
    title_col, fd_col = read_data(file_name)
    docsAsShingleSets = shingle_data()
    write_to_file(output_file, docsAsShingleSets)
    # else:
    #     docsAsShingleSets = read_pkl(output_file)
    #     print docsAsShingleSets

    CreateSignature()


