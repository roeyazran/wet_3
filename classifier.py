data_path = r'data/Data.pickle'
import pickle
import numpy as np
from sklearn import *
from hw3_utils import *
from sklearn.linear_model import Perceptron
import random
import csv

GlobalDataSet = load_data(data_path);

def euclidean_distance(ObjectA, ObjectB):
    distance = 0
    #print ("euclidean dist, lists lens:",len(ObjectA), ", " , len(ObjectB))
    for index in range(len(ObjectA)):
        distance += (ObjectA[index]-ObjectB[index])**2
    return distance**(0.5)



class knn_classifier(abstract_classifier):
    def __init__(self, DataFeatures, DataLabels, k=1):
        self.k = k
        self.DataFeatures = DataFeatures
        self.DataLabels = DataLabels

    def classify(self, features):
        ObjectsDistances = np.empty(len(self.DataFeatures))
        ObjectsLabels = np.empty(len(self.DataLabels))
        for index in range(len(self.DataFeatures)):
            ObjectsDistances[index] = euclidean_distance(self.DataFeatures[index], features)
            #print ("distance: ", ObjectsDistances[index])
            ObjectsLabels[index] = self.DataLabels[index]
        countPos = 0
        countNeg = 0
        #counting the classification of the closest k items
        for index in ObjectsDistances.argsort()[:self.k]:
            #print (index," closest distance: ", ObjectsDistances[index])
            if ObjectsLabels[index]:
                countPos += 1
            else:
                countNeg += 1
        return True if countPos >= countNeg else False

class knn_factory(abstract_classifier_factory):

    def __init__(self, k=1):
        self.k = k

    def train(self, data, labels):
        #print("data:", len(data), " labels: ", len(labels))
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
            NegList = removeElementFromList(NegList, ElementToAdd)
            if len(NegList) == 0:
                break

    for index in range(1, num_folds + 1):
        pickle.dump((ECGFoldLists[index-1], ECGFoldLabels[index-1]), open("ecg_fold_"+str(index)+".data", "wb"))



def load_k_fold_data(index):
    with open('ecg_fold_' + str(index) + '.data','rb') as f:
        train_index, labels_index = pickle.load(f)
    return train_index, labels_index

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
    for index in range(k):
        print ("Testing ", index + 1, " fold")
        TraningSetData = list()
        TraningSetLabels = list()
        for j in range(k):
            if j != index:
                #print(foldsLists[j][0])
                TraningSetLabels.extend(foldsLists[j][1])
                TraningSetData.extend(foldsLists[j][0])
        Classifier = classifier_factory.train(TraningSetData, TraningSetLabels)
        for SampleIndex, sample in enumerate(foldsLists[index][0]):
            label = foldsLists[index][1][SampleIndex]
            ClassRes = Classifier.classify(sample)
            #print (label)
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
    print("OverAll results folds= ", k, "AvgAccuracy= ", AvgAccuracy, "AvgError= ", AvgError )
    return AvgAccuracy, AvgError


def experiments6():
    with open('experiments6.csv', mode= 'w') as experiments6_file:
        exp6_writer= csv.writer(experiments6_file, delimiter=',', quotechar='"', quoting = csv.QUOTE_MINIMAL)
        for k in [1, 3, 5, 7, 13]:
            print("Testing for k = ",k)
            ClassifierFactory = knn_factory(k)
            acc, err = evalute(ClassifierFactory, 2)
            exp6_writer.writerow([k, acc, err])


class DIClassifier(tree.DecisionTreeClassifier):
    def classify(self, features):
        return self.predict([features])[0]

class DI_factory(abstract_classifier_factory):
    def train(self, data, labels):
        #print("data:", len(data), " labels: ", len(labels))
        Classifier = DIClassifier()
        return Classifier.fit(data, labels)


class PerceptronFactory(abstract_classifier_factory):
    def train(self, data, labels):
        #print("data:", len(data), " labels: ", len(labels))
        Classifier = PerceptronClassifier()
        return Classifier.fit(data, labels)


class PerceptronClassifier(Perceptron):
    def classify(self, features):
        return self.predict([features])[0]


def experiment1():
    with open('experiments12.csv', mode= 'w') as experiments1_file:
        exp1_writer= csv.writer(experiments1_file, delimiter=',', quotechar='"', quoting = csv.QUOTE_MINIMAL)
        ClassifierFactory = DI_factory()
        acc, err = evalute(ClassifierFactory, 2)
        exp1_writer.writerow(["1", acc, err])
        ClassifierFactory= PerceptronFactory()
        acc, err = evalute(ClassifierFactory, 2)
        exp1_writer.writerow(["2", acc, err])

if __name__ == '__main__':
#    print(GlobalDataSet)
#    split_crosscheck_groups(GlobalDataSet, 2)
    experiments6()
#    experiment1()

