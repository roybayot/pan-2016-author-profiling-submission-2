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


from os import listdir
from os.path import isfile, join

reload(sys)
sys.setdefaultencoding("ISO-8859-1")


def getRelevantDirectories(argv):
    inputDir = ''
    outputDir = ''
    modelDir = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:m:",["ifile=","ofile=","mfile="])
    except getopt.GetoptError:
        print './myTestingSoftware.py -i <inputdirectory> -m <modelfile> -o <outputdirectory>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print './myTestingSoftware.py -i <inputdirectory> -m <modelfile> -o <outputdirectory>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputDir = arg
        elif opt in ("-m", "--mfile"):
            modelDir = arg
        elif opt in ("-o", "--ofile"):
            outputDir = arg
    return inputDir, outputDir, modelDir

def isPKL(fileName):
    a = fileName.strip().split('.')
    if a[1] == 'pkl':
        return True
    else:
        return False
	
def getAllModels(modelDir):
# 	only_files = [ f for f in listdir(modelDir) if isfile(join(modelDir,f)) ]
    allFiles = absoluteFilePaths(modelDir)
    pklFiles = [f for f in allFiles if isPKL(join(modelDir,f))]
    pklFiles = [f for f in pklFiles if isNotVectorizer(join(modelDir,f))]
    
    models = {}
    oneFile = pklFiles[0]
    f = open(oneFile, 'rb')
    oneModel = pickle.load(f)
    f.close()
    
    return oneModel


def dirExists(inputDir):
    if os.path.exists(inputDir):
        return True
    elif os.access(os.path.dirname(inputDir), os.W_OK):
        print "Cannot access the directory. Check for privileges."
        return False
    else:
        print "Directory does not exist."
        return False

def isXML(f):
    a = f.strip().split('.')
    if a[1] == 'xml':
        return True
    else:
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

def getAllTestFiles(inputDir):
    if dirExists(inputDir):
        allTestFiles = [ f for f in listdir(inputDir) if isfile(join(inputDir,f)) ]
        allTestFiles = [ f for f in allTestFiles if isXML(f) ]
		return allTestFiles
    else:
        sys.exit()

def getAllXmlFiles(allTestFiles):
    allTestFiles = [ f for f in allTestFiles if isfile(f) ]
    allTestFiles = [ f for f in allTestFiles if isXML(f) ]
    return allTestFiles	

def getLanguage(oneFile):
    tree = ET.parse(oneFile)
    root = tree.getroot()
    a = root.attrib
    return a['lang']

def getTweetsToLine(oneFile):
    allText = ""
    try:
        tree = ET.parse(oneFile)
        print "Filename: %s SUCCESS!" % oneFile
    except:
        e = sys.exc_info()[0]
        print "Filename: %s Error: %s" % (oneFile, e)
    else:
        root = tree.getroot()
        a = []
        
        for x in root.iter("document"):
            a.append(x.text)
            
        allText = ""
        
        for doc in a:
            clean = bleach.clean(doc, tags=[], strip=True)
            allText = allText + clean
        allText = allText.encode('utf-8')«
    return allText

def clean_text(raw_text):
    review_text = BeautifulSoup(raw_text).get_text()
    words = review_text.lower().split()
    return(" ".join(words))

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


 	
def classifyTestFiles(models, inputDir):
#    current_working_dir = os.getcwd() + '/'
    current_working_dir = './'
    model_dir = "word2vec-models/wikipedia-only-trained-on-my-machine/"
    num_features = 100
    tasks =["age", "gender"]

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
    results = {}
    allTestFiles = getAllFilenamesWithAbsPath(inputDir)
    allTestFiles = getAllXmlFiles(allTestFiles)
    
    lingua = getLanguage(allTestFiles[0])
#    import ipdb; ipdb.set_trace()
    if lingua == 'EN':
        tempLang = 'english'
	if lingua == 'NL':
		tempLang = 'dutch'
	if lingua == 'ES':
		tempLang = 'spanish'

    filename = relations[tempLang]['model_file']
    filename = current_working_dir + model_dir + filename 
    word2vec_model = gensim.models.Word2Vec.load(filename) 
    
    tasks = ["gender", "age"]

    for oneFile in allTestFiles:
        lang = getLanguage(oneFile)
        aa = oneFile.strip().split("/")
        aa = aa[-1].strip().split(".")
        
        thisId					= aa[0]
        thisType				= 'not relevant'
        thisLanguage			= lang
        
        oneLine = getTweetsToLine(oneFile)
        oneLine = clean_text(oneLine)
        
        temp = {}
        temp['thisId']       = thisId
        temp['thisType']     = thisType
        temp['thisLanguage'] = thisLanguage
        descriptors, trash = makeFeatureVec(oneLine, word2vec_model, num_features)

        for task in tasks:
            key_name = tempLang + "_" + task
            
            if tempLang == "dutch" and task == "age":
                pred_value = 5
            else:
                clf = models[key_name]
                pred_value = clf.predict(descriptors.reshape(1,-1))

            if task == "age":
                if pred_value == 0:
                    temp['age'] = '18-24'
                elif pred_value == 1:
                    temp['age'] = '25-34'
                elif pred_value == 2:
                    temp['age'] = '35-49'
                elif pred_value == 3:
                    temp['age'] = '50-64'
                elif pred_value == 4:
                    temp['age'] = '65-xx'
                else:
                    temp['age'] = 'XX-XX'
            else:
                if pred_value == 0:
                    temp['gender'] = 'male'
                else:
                    temp['gender'] = 'female'
        results[oneFile] =  temp
    return results
	
def writeOneResult(key, value, outputDir):
	key = key.strip().split("/")
	cwd = os.getcwd()
	path = cwd + "/" + outputDir + "/" + key[-1]
# 	import pdb; pdb.set_trace()
	thisId					= value['thisId']
	thisType				= value['thisType']
	thisLanguage			= value['thisLanguage']
	predictedGender 	 	= value['gender']
	predictedAge    	 	= value['age']

	
	text_to_write = """<author id='%s'\n\ttype='%s'\n\tlang='%s'\n\tage_group='%s'\n\tgender='%s'\n/>"""% (thisId, thisType, thisLanguage, predictedAge, predictedGender)
	# Open a file
	fo = open(path, "w")
	fo.write( text_to_write );
	fo.close()
	
def makeDirectory(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
        else:
            print "\nBE CAREFUL! Directory %s already exists." % path
		
def writeAllResults(results, outputDir):
    if (not dirExists(outputDir)):
        print "Creating new directory."
        makeDirectory(outputDir)
    for key, value in results.iteritems():
        writeOneResult(key, value, outputDir)	

def getLang(inputText):
    langs = ["english", "dutch", "spanish"]
    lang = [ lang for lang in langs if lang in inputText]
    return lang[0]
	
def main(argv):
    inputDir, outputDir, modelDir = getRelevantDirectories(argv)
    print 'Input directory is "',  inputDir
    print 'Model directory is "',  modelDir   
    print 'Output directory is "', outputDir
    
    models = getAllModels(modelDir)
    results = classifyTestFiles(models, inputDir)
    writeAllResults(results, outputDir)

   
if __name__ == "__main__":
    main(sys.argv[1:])
