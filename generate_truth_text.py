#!/usr/bin/env python
#!/home/darklord/anaconda2/bin/python

import sys
import getopt
import bleach
import xml.etree.ElementTree as ET
import os
import re
import csv
import pickle


import pandas as pd
import numpy as np
import re
import timeit

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC

from sklearn import cross_validation
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from bs4 import BeautifulSoup


reload(sys)
sys.setdefaultencoding("ISO-8859-1")
#sys.setdefaultencoding("UTF-8")
#sys.setdefaultencoding("latin-1")


def dirExists(inputDir):
    if os.path.exists(inputDir):
        return True
    elif os.access(os.path.dirname(inputDir), os.W_OK):
        print "Cannot access the directory. Check for privileges."
        return False
    else:
        print "Directory does not exist."
        return False

def absoluteFilePaths(directory):
    allPaths = []
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            onePath = os.path.abspath(os.path.join(dirpath, f))
            allPaths.append(onePath)
    return allPaths

def getAllFilenamesWithAbsPath(inputDir):
    if dirExists(inputDir):
        allPaths = absoluteFilePaths(inputDir)
        return allPaths
    else:
        sys.exit()

def isTruthTextFile(f):
    return 'truth.txt' in f

def getTruthTextFiles(allPaths):
    return [f for f in allPaths if isTruthTextFile(f)]



def getRelevantDirectories(argv):
    inputDir = ''
    outputDir = ''
    modelDir = ''
    
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print './myTrainingSoftware.py -i <inputdirectory> -o <outputdirectory>'
        print 'The input directory should contain all the training files. \nThe output directory will be where the models are stored.'
        sys.exit(2)
    
    for opt, arg in opts:
        if opt == '-h':
            print './myTrainingSoftware.py -i <inputdirectory> -o <outputdirectory>'
            print 'The input directory should contain all the training files. \nThe output directory will be where the models are stored.'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputDir = arg
        elif opt in ("-o", "--ofile"):
            outputDir = arg
    return inputDir, outputDir


def writeOneSummary(outputFilename, oneTruthFile, allPaths):
    data = ["filename", "gender", "age", "text"]
    path = outputFilename.strip().split("/")
    outputFilename = path[-1]
    print "Output filename: ", outputFilename

    path = '/'.join(path[0:-1])

    tsv_writer(data, outputFilename)
    gender = {'MALE': 0, 'FEMALE':1}
    ageGroup = {'18-24': 0, \
                '25-34': 1, \
                '35-49': 2, \
                '50-64': 3, \
                '65-xx': 4, \
                'XX-XX': None}

    one_file = open(oneTruthFile, 'r')

    for line in one_file:
        a = line.strip().split(":::")
        fileName 		  = path+ "/" + a[0] + ".xml"
# 		print fileName
        thisGender 	 	  = gender[a[1]]
        thisAgeGroup 	  = ageGroup[a[2]]

        try:
            tree = ET.parse(fileName)
# 			print "Filename: %s SUCCESS!" % fileName

        except:
            e = sys.exc_info()[0]
            print "Filename: %s Error: %s" % (fileName, e)
        else:
            root = tree.getroot()
            a = []
            for x in root.iter("document"):
                a.append(x.text)

# 			print "In Else"
            allText = ""

# 			print "Going in for loop"
           
            for doc in a:
                clean = bleach.clean(doc, tags=[], strip=True)
                allText = allText + clean
            #allText = allText.encode('ISO-8859-1')
            allText = allText.encode('utf-8')
            #allText = allText.encode('latin-1')
            #allText = allText.replace("\"", " ")
            #allText = allText.replace("...", " ")
# 			print "Out of loop, writing"								
            data = [fileName, thisGender, thisAgeGroup, allText]
            tsv_writer(data, outputFilename)
# 			print "Finish writing one line"

def tsv_writer(data, path):
    """ Write data to a TSV file path """
    with open(path, "a") as tsv_file:
        writer = csv.writer(tsv_file, delimiter='\t')
        writer.writerow(data)


def deriveOutputFilename(oneTruthFile, langs):
    a = oneTruthFile.strip().split("/")
    lang = [ lang for lang in langs if lang in oneTruthFile]
    print "Processing: ", lang[0]

    outputFilename = '/'.join(a[0:-1]) + '/summary-' + lang[0] + '-' + a[-1]
    return outputFilename

def generateTruthTexts(allPaths, allTruthText, outputDir, langs):
    for oneTruthFile in allTruthText:
        outputFilename = deriveOutputFilename(oneTruthFile, langs)
        writeOneSummary(outputFilename, oneTruthFile, allPaths)





def main(argv):
    inputDir, outputDir = getRelevantDirectories(argv)
    print "input dir:", inputDir
    print "output dir:", outputDir

    allPaths = getAllFilenamesWithAbsPath(inputDir)

    allTruthText = getTruthTextFiles(allPaths)
    langs=["english", "dutch", "spanish"]
    generateTruthTexts(allPaths, allTruthText, outputDir, langs)

if __name__ == "__main__":
    main(sys.argv[1:])
