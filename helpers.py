import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
import sys
import os

import sklearn
from sklearn import cross_validation
from sklearn import svm
from sklearn import metrics
from scipy import stats

import re

reload(sys)
#sys.setdefaultencoding("ISO-8859-1")
sys.setdefaultencoding("UTF-8")


def clean_text(raw_text):
    review_text = BeautifulSoup(raw_text).get_text()
    words = review_text.lower().split()
    return(" ".join(words))

def clean_all_text(allText, numLines):
    clean_train_data = []
    for i in xrange(0, numLines):
        clean_train_data.append(clean_text(allText[i]))
    return clean_train_data

def getMeanAccuracies(results_dictionary):
    accuracies_dictionary = {}
    for each_key in results_dictionary.keys():
        list_of_accuracies = results_dictionary[each_key]
        accuracies = [a.mean() for a in list_of_accuracies]
        accuracies_dictionary[each_key] = accuracies
    return accuracies_dictionary

def getSortedKeys(all_results):
    sorted_keys = all_results.keys()
    sorted_keys.sort()
    return sorted_keys


def turnMeanAccuraciesToExcel(file_name, results_dictionary):
    accuracies_dictionary = getMeanAccuracies(results_dictionary)
    list_of_accuracy_settings = getListOfSettings(accuracies_dictionary)
    i = len(list_of_accuracies)
    accuracy_values = np.zeros(1,i)
    for x in range(i):
        accuracy_values[x] = accuracies_dictionary[list_of_accuracy_settings[x]]
    df = pd.DataFrame(data=accuracy_values, columns=list_of_accuracy_settings)
    df.index="accuracy"
    df.to_csv(file_name, sep=',', encoding='utf-8')
    
def makePValMatrix(all_results):
    sorted_keys = getSortedKeys(all_results)
    list_length = len(sorted_keys)
    p_value_matrix = np.zeros((list_length, list_length))
    i = range(0, list_length)
    #sig values
                            
    for key_1, x in zip(sorted_keys, i):
        for key_2, y in zip(sorted_keys, i):
            treatment_1 = all_results[key_1]
            treatment_2 = all_results[key_2]
            z_stat, p_val = stats.ranksums(treatment_1, treatment_2)
            p_value_matrix[x,y] = p_val
    
    return p_value_matrix

def turnPValMatrixToExcel(fileName, all_results):
    p_value_matrix = makePValMatrix(all_results)
    sorted_keys = getSortedKeys(all_results)
    df = pd.DataFrame(data = p_value_matrix, columns=sorted_keys)
    df.index = sorted_keys
    null_disproved = df[df < 0.05]
    null_disproved.to_csv(fileName, sep=',', encoding='utf-8')


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
        reviewFeatureVecs[counter], wordsNotInDict = makeFeatureVec(one_line, model, num_features)
        lineOfWordsNotInDict.append(wordsNotInDict)
        counter = counter + 1.
        
    return reviewFeatureVecs, lineOfWordsNotInDict

def doSVMwithPoly(trainDataVecs, targetVec, source, num_features, task,\
        num_folds=10, degrees=[1,2,3], C=[10**-1, 10, 10**3],\
        scoring_function="accuracy"):
    
    poly_results = {}
    for degree in degrees:
        for one_C in C:
            clf = svm.SVC(kernel='poly', degree=degree, coef0=one_C, gamma=1)
            scores = cross_validation.cross_val_score(clf, trainDataVecs,\
                                                      targetVec, cv=num_folds,\
                                                      scoring=scoring_function)

            string_pattern = "word2vec-source={} dims={} task={} kernel={} degree={} C={}"
                                                                        
            dict_key = string_pattern.format(source, num_features, task, \
                                             "poly", degree, one_C)
            poly_results[dict_key] = scores
    return poly_results


def doSVMwithRBF(trainDataVecs, targetVec, source, num_features, task,\
                 num_folds=10, gammas=[1, 0.001], C = [10, 1000],\
                 scoring_function="accuracy"):
   
    rbf_results = {}
    for g in gammas:
        for one_C in C:
            clf = svm.SVC(kernel='rbf', gamma=g, C=one_C)
            scores = cross_validation.cross_val_score(clf, trainDataVecs,\
                                                      targetVec, cv=10,\
                                                      scoring=scoring_function)
            
            string_pattern = "word2vec-source={} dims={} task={} kernel={} gamma={} C={}"
            dict_key = string_pattern.format(source, num_features, task, \
                                            "rbf",g, one_C)
            rbf_results[dict_key] = scores
    
    return rbf_results


def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z
