# coding=utf8
import csv
import numpy as np
import itertools
import nltk
import io

def getSentenceData(path, vocabulary_size=8000):
    print("Reading CSV file...")

    with open(path, 'r') as f:
        cnt = 0
        for line in f:
            line = line.decode('utf8').encode()
            print line
            cnt += 1
            if cnt > 50:
                break

if __name__ == '__main__':
    getSentenceData('data/reddit-comments-2015-08.csv')
