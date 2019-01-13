data_path = r'data/Data.pickle'
import pickle
import numpy as np
import sklearn
from hw3_utils import *
import random
import csv

GlobalDataSet = load_data(data_path);

def euclidean_distance(ObjectA, ObjectB):
    distance = 0
    for (featureA1,featureA2) in (ObjectA, ObjectB):
        distance += (featureA1-featureA2)**2
    return distance**(0.5)



class knn_classifier(abstract_classifier):
    def __init__(self, DataFeatures, DataLabels, k=1):
        self.k = k
        self.DataFeatures = DataFeatures
        self.DataLabels = DataLabels

    def classify(self, features):
        ObjectsDistances = np.empty(len(self.DataFeatures))
        ObjectsLabels = np.empty(len(self.DataLabels))
        for index, (object, label) in enumrate((self.DataFeatures,self.DataLabels)):
            ObjectsDistances[index]=euclidean_distance(object, features)
            ObjectsLabels[index]=label
        countPos = 0
        countNeg = 0
        for index in ObjectsDistances.argsort()[:self.k]:
            if ObjectsLabels[index]:
                countPos += 1
            else:
                countNeg += 1
        return True if countPos >= countNeg else False

class knn_factory(abstract_classifier_factory):

    def __init__(self, k=1):
        self.k = k

    def train(self, data, labels):
        return knn_classifier(data,labels, self.k)


def removeElementFromList(givenList,element):
    foundElement = False
    tempPosList = list()
    for x in givenList:
        if (x == element).all() and foundElement == False:
            foundElement = True
        else:
            tempPosList.append(x)
    return tempPosList

def split_crosscheck_groups (dataset, num_folds):
    PosList = [dataset[0][i] for i,x in enumerate(dataset[1]) if x == True]
    NegList = [dataset[0][i] for i,x in enumerate(dataset[1]) if x == False]
    ECGFoldLists = list()
    ECGFoldLabels = list()

    for i in range (num_folds):
        ECGFoldLists.append(list())
        ECGFoldLabels.append(list())

    while len(PosList) != 0:
        for index in range(num_folds):
            ElementToAdd = random.choice(PosList)
            ECGFoldLists[index].append(ElementToAdd)
            ECGFoldLabels[index].append(True)
            PosList = removeElementFromList(PosList,ElementToAdd)
            if len(PosList) == 0 :
                break

    while len(NegList) != 0:
        for index in range(num_folds):
            ElementToAdd = random.choice(NegList)
            ECGFoldLists[index].append(ElementToAdd)
            ECGFoldLabels[index].append(False)
            NegList = removeElementFromList(NegList,ElementToAdd)
            if len(NegList) == 0:
                break

    for index in range(1, num_folds + 1):
        pickle.dump((ECGFoldLists[index-1], ECGFoldLabels[index-1]), open("ecg_fold_"+str(index)+".data", "wb"))



def load_k_fold_data(index):
    with open('ecg_fold_' + str(index) + '.data','rb') as f:
        train_index, labels_index = pickle.load(f)
    return train_index, labels_index

if __name__ == '__main__':
#    print(GlobalDataSet)
#    split_crosscheck_groups(GlobalDataSet, 2)
    load_k_fold_data(2)

def evalute(classifier_factory, k):
    foldsLists = list()
    DatasetLen = 0
    for index in range(1, k+1):
        foldsLists.append(load_k_fold_data(index))
        DatasetLen += len(load_k_fold_data(index)[0])

    TruePos = 0
    TrueNeg = 0
    FalsePos = 0
    FalseNeg = 0
    for index in range(1, k+1):
        TraningSetData = list()
        TraningSetLabels = list()
        for j in range(1, k + 1):
            if j != index:
                TraningSetLabels.append(foldsLists[j][1])
                TraningSetData.append(foldsLists[j][0])
        Classifier = classifier_factory.train(TraningSetData, TraningSetLabels)
        for sample,label in foldsLists[index][0],foldsLists[index][1]:
            ClassRes = Classifier.classify(sample)
            if label == True:
                if ClassRes == True:
                    TruePos += 1
                else:
                    FalseNeg += 1
            else:
                if ClassRes == False:
                    TrueNeg += 1
                else:
                    FalsePos += 1

        AvgAccuracy = (TruePos+TrueNeg)/DatasetLen
        AvgError = (FalsePos + FalseNeg)/DatasetLen
        return AvgAccuracy, AvgError


def experiments6():
    with open('experiments6.csv', mode= 'w') as experiments6_file:
        exp6_writer= csv.writer(experiments6_file, delimiter=',', qutechar='"', quoting = csv.QUOTE_MINIMAL)
        for k in [1, 3, 5, 7, 13]:
            ClassifierFactory = knn_factory(k)
            acc,err = evalute(ClassifierFactory, 2)
            exp6_writer.writerow([k, acc, err])



