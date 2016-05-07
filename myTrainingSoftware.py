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
import gensim

from sklearn import cross_validation
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from bs4 import BeautifulSoup

import helpers


reload(sys)
sys.setdefaultencoding("ISO-8859-1")
#sys.setdefaultencoding("UTF-8")
#sys.setdefaultencoding("latin-1")

def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.
    index2word_set = set(model.index2word)
    wordsNotInDict = []

    try:
        words = words.split(" ")
    except AttributeError:
        featureVec = np.random.rand(num_features)
        return featureVec, wordsNotInDict

    
    for word in words:
        if word in index2word_set:
            #import ipdb; ipdb.set_trace()
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
        else:
            wordsNotInDict.append(word)
            
    if nwords == 0.:
        featureVec = np.random.rand(num_features)
    else:
        featureVec = np.divide(featureVec,nwords)
    return featureVec, wordsNotInDict

def getAvgFeatureVecs(all_texts, model, num_features):
    counter = 0.
    reviewFeatureVecs = np.zeros((len(all_texts),num_features),dtype="float32")
    lineOfWordsNotInDict = []
    
    for one_line in all_texts:
        #print "============"+ str(counter)+"============="
        reviewFeatureVecs[counter], wordsNotInDict = makeFeatureVec(one_line, model, num_features)
        lineOfWordsNotInDict.append(wordsNotInDict)
        #print "============"+str(counter)+"============="
        counter = counter + 1.
        
    return reviewFeatureVecs, lineOfWordsNotInDict



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


def writeOneSummary(outputFilename, oneTruthFile, allPaths, realOutputFilename):
    data = ["filename", "gender", "age", "text"]
    path = outputFilename.strip().split("/")
    outputFilename = path[-1]
    print "Output filename: ", outputFilename

    path = '/'.join(path[0:-1])

    tsv_writer(data, realOutputFilename)
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
        
        parser = helpers.MyXMLParser(encoding='utf-8')

        try:
            tree = ET.parse(fileName, parser=parser)
            #tree = ET.parse(fileName)
            #print "Filename: %s SUCCESS!" % fileName

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
            tsv_writer(data, realOutputFilename)
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
    #outputFilename = 'summary-' + lang[0] + '-' + a[-1]
    return outputFilename

def generateTruthTexts(allPaths, allTruthText, outputDir, langs):
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    
    for oneTruthFile in allTruthText:
        outputFilename = deriveOutputFilename(oneTruthFile, langs)

        some_fill_in = outputFilename.split("/")

        print some_fill_in[-1]
        #realOutputFilename = os.getcwd() + '/' + outputDir + '/' + some_fill_in[-1] 
        realOutputFilename = outputDir + '/' + some_fill_in[-1] 
        print outputFilename
        writeOneSummary(outputFilename, oneTruthFile, allPaths, realOutputFilename)

def trainOne(X,y,lang,task):
    if lang == "english" and task == "age":
        clf = svm.SVC(kernel='rbf', gamma=100, C=100)
        clf.fit(X, y)
    if lang == "english" and task == "gender":
        clf = svm.SVC(kernel='rbf', gamma=100, C=100)
        clf.fit(X, y)
    if lang == "dutch" and task == "age":
        clf = [] 
    if lang == "dutch" and task == "gender":
        clf = svm.SVC(kernel='rbf', gamma=1, C=100)
        clf.fit(X, y)
    if lang == "spanish" and task == "age":
        clf = svm.SVC(kernel='rbf', gamma=1, C=100)
        clf.fit(X, y)
    if lang == "spanish" and task == "gender":
        clf = svm.SVC(kernel='rbf', gamma=1, C=100)
        clf.fit(X, y)
    
    return clf

def writeModels(models, outputDir):
    fileName = outputDir + "/models.pkl"
    f = open(fileName, 'wb')
    pickle.dump(models, f)
    f.close()


def main(argv):
    num_features = 100
    langs=["english", "dutch", "spanish"]
    tasks=["age", "gender"]
    scoring_function = "accuracy"
#    current_working_dir = os.getcwd() + '/'
    current_working_dir = './'
    model_dir = "word2vec-models/wikipedia-only-trained-on-my-machine/"

    classification_models = {}
    
    relations = {'dutch': {'truth_file': 'summary-dutch-truth.txt',\
                        'model_file': 'wiki.nl.tex.d100.model'
                        },
                'english': {'truth_file': 'summary-english-truth.txt',\
                            'model_file': 'wiki.en.tex.d100.model'
                            },
                'spanish': {'truth_file': 'summary-spanish-truth.txt',\
                            'model_file': 'wiki.es.tex.d100.model'
                            }
                }
    inputDir, outputDir = getRelevantDirectories(argv)
    print "INPUT DIR", inputDir
    print "OUTPUT DIR", outputDir 
    allPaths = getAllFilenamesWithAbsPath(inputDir)
    allTruthText = getTruthTextFiles(allPaths)
    print "ALL TRUTH TEXT", allTruthText
    generateTruthTexts(allPaths, allTruthText, outputDir, langs)

    modelDir = os.getcwd() + '/' + outputDir 

    for f in allTruthText:
        a = f.strip().split("/")
        lang = [ lang for lang in langs if lang in f]
        print "Processing: ", lang[0]
        lang = lang[0]

        truth_file = relations[lang]['truth_file']
        model_file = current_working_dir + model_dir + relations[lang]['model_file']

        #truth_file = current_working_dir + outputDir + "/" + truth_file
        truth_file = outputDir + "/" + truth_file
        train = pd.read_csv(truth_file, header=0, delimiter="\t", quoting=1)
        print "Done reading file"
        clean_train_data = train['text']
        
        model = gensim.models.Word2Vec.load(model_file)
        
        trainDataVecs, trashedWords = getAvgFeatureVecs( clean_train_data,\
                                                         model,\
                                                         num_features )
        X = trainDataVecs

        for task in tasks:
            key_name = lang + "_" + task
            
            if lang == "dutch" and task == "age":
                classification_models[key_name] = []
            else:
                y = train[task]
                one_model = trainOne(X, y, lang, task)
                classification_models[key_name] = one_model

    writeModels(classification_models, modelDir)

if __name__ == "__main__":
    main(sys.argv[1:])
