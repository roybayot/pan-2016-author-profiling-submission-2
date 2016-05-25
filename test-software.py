import gensim
import pandas as pd
import numpy as np
import sys
import os
import itertools
import sklearn

from sklearn import cross_validation
from sklearn import svm
from sklearn import metrics
from scipy import stats
from bs4 import BeautifulSoup


import helpers as helper
import pickle

import sys
reload(sys)
#sys.setdefaultencoding("ISO-8859-1")
sys.setdefaultencoding("UTF-8")

"""This will use trained word2vec wikipedia models on dutch, english, spanish.
It will use word2vec with dimension 100."""

def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.
    index2word_set = set(model.index2word)
    wordsNotInDict = []

    #words = words.strip()

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


current_working_dir = os.getcwd() + '/'
model_dir = "word2vec-models/wikipedia-only-trained-on-my-machine/"
#model_dir = "word2vec-models/glove-twitter/"
#model_dir = "word2vec-models/wiki-giga/"


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
#relations = {'dutch': {'truth_file': 'summary-dutch-truth.txt',\
#                       'model_file': 'wiki.nl.tex.d100.model'
#                       },
#             'english': {'truth_file': 'summary-english-truth.txt',\
#                         'model_file': 'new.glove.twitter.27B.200d.txt'
#                        },
#             'spanish': {'truth_file': 'summary-spanish-truth.txt',\
#                         'model_file': 'wiki.es.tex.d100.model'
#                        }
#             }
#relations = {'dutch': {'truth_file': 'summary-dutch-truth.txt',\
#                       'model_file': 'wiki.nl.tex.d100.model'
#                       },
#             'english': {'truth_file': 'summary-english-truth.txt',\
#                         'model_file': 'new.glove.6B.100d.txt'
#                        },
#             'spanish': {'truth_file': 'summary-spanish-truth.txt',\
#                         'model_file': 'wiki.es.tex.d100.model'
#                        }
#             }
#

tasks = ['age', 'gender']
num_features = 100
#num_features = 200
source = "wikipedia-self-trained"
poly_degrees = [1, 2, 3]
poly_C = [10**-2, 1, 10**2]
rbf_gammas = [10**-2, 1, 10**2]
rbf_C =[10**-2, 1, 10**2]


all_results = {}


#for lang in relations.keys():

#for lang in ["english", "spanish"]:
for lang in ["spanish"]:
    scoring_function = "accuracy"
    truth_file = relations[lang]['truth_file']
    model_file = current_working_dir + model_dir + relations[lang]['model_file']

    train = pd.read_csv(truth_file, header=0, delimiter="\t", quoting=1)
    print "Done reading file"
    #clean_train_data = helper.clean_all_text(train["text"], train["text"].size)
    clean_train_data = train['text']

    #model = pickle.load( open( model_file, "rb" ) )
    model = gensim.models.Word2Vec.load(model_file)
    #model = gensim.models.Word2Vec.load_word2vec_format(model_file,binary=False)

    #trainDataVecs, trashedWords = helper.getAvgFeatureVecs( clean_train_data,\
    #                                                        model,\
    #                                                        num_features )
    trainDataVecs, trashedWords = getAvgFeatureVecs( clean_train_data,\
                                                            model,\
                                                            num_features )

    print "Done making average vector"
    
    for task in tasks:
        print "================="
        print lang, task
        print "================="
        if lang == "dutch"  and task == "age":
            pass
        else:
            train_y = train[task]
            train_y = np.array(train_y)
            poly_results = helper.doSVMwithPoly(trainDataVecs, train_y, source, \
                                            num_features, task, num_folds=10,\
                                            degrees=poly_degrees, C=poly_C,\
                                            scoring_function="accuracy")
            print "Done with polynomial", task
            rbf_results = helper.doSVMwithRBF(trainDataVecs, train_y, source, \
                                            num_features, task, num_folds=10,\
                                            gammas=rbf_gammas, C=rbf_C,\
                                            scoring_function="accuracy")

            print "Done with rbf", task

            results_one_task = helper.merge_two_dicts(poly_results, rbf_results)
            all_results = helper.merge_two_dicts(results_one_task, all_results)

sorted_keys = all_results.keys()
sorted_keys.sort()

for key in sorted_keys:
    print key, all_results[key].mean()

