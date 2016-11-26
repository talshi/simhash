
import pandas as pd
import binascii
import random
import time

# constants
maxShingleID = 2 ** 32 - 1
nextPrime = 4294967311
numHashes = 10

docsNames = []
title_col = []
fd_col = []
docsAsShingleSets = {};



# read data file
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

        shinglesInDoc = set()

        shingle_list = [doc[i:i + 5] for i in range(len(doc) - 5 + 1)]

        # shinglesInDoc = set(binascii.crc32(single_shingle) for single_shingle in shingle_list)

        for single_shingle in shingle_list:
            crc = binascii.crc32(single_shingle) & 0xffffffff
            shinglesInDoc.add(crc)

        docsAsShingleSets[docID] = shinglesInDoc

    return docsAsShingleSets

def rand_coeffs(numHashes):
    randList = []

    k = numHashes

    while k > 0:
        # Get a random shingle ID.
        randIndex = random.randint(0, maxShingleID)

        # Ensure that each random number is unique.
        while randIndex in randList:
            randIndex = random.randint(0, maxShingleID)

            # Add the random number to the list.
        randList.append(randIndex)
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

            minHashCode = nextPrime + 1 # ??

            for shingleID in shingleIDSet:

                # (A*x+B) % p
                hashCode = (A[i] * shingleID + B[i]) % nextPrime

                if hashCode < minHashCode:
                    minHashCode = hashCode

            signature.append(minHashCode)

        signatures.append(signature)

    t = time.time() - t_delta

    print 'creating signatures took %.2fsec' % t

def jacard():
    pass

if __name__ == "__main__":
    file_name = 'Test2.csv'
    title_col, fd_col = read_data(file_name)

    docsAsShingleSets = shingle_data()

    CreateSignature()


