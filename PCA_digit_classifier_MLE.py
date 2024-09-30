from multiprocessing.dummy import Array
import scipy as sp
import numpy as np 
from scipy.io import loadmat
from scipy.stats import multivariate_normal, mode
from sklearn.decomposition import PCA
from scipy.spatial import KDTree
import matplotlib.pyplot as plt




data  =  loadmat('digits.mat')

class mleClassifier():
    def __init__(self):
        self.rawData  =  data
        self.classPriors = dict()
        self.classConditionalData = dict()
        self.classConditionals  = dict()
        self.X = None
        self.Y = None
        self.correctlyClassifed = 0
        self.accuracy = None

    def processData(self):
        self.X =  self.returnPrincipalComponents(self.rawData['X']).T
        self.Y = self.rawData['Y']  


    def returnPrincipalComponents(self, X_data):
        pca  = PCA(n_components=20)
        pca.fit(X_data.T)
        return pca.components_

    def splitTrainingTesting(self, split):
        entries =  len(self.X)
        trainingDataX  = self.X[0:int(entries*split)]
        testingDataX = self.X[int(entries*split):entries]

        trainingDataY  = self.Y[0:int(entries*split)]
        testingDataY = self.Y[int(entries*split):entries]

        self.trainingDataX =  trainingDataX
        self.trainingDataY =  trainingDataY 

        self.testingDataX =  testingDataX 
        self.testingDataY =  testingDataY 

        return (trainingDataX, trainingDataY), (testingDataX, testingDataY)
        
    def computeClassPriors(self):
        totalCount  = len(self.trainingDataY)
        for digitClassification in self.trainingDataY:
            digit = digitClassification[0]
            if(digit not in self.classPriors):
                self.classPriors[digit] = 1/totalCount
            else:
                self.classPriors[digit] = self.classPriors[digit]+1/totalCount
        

    def computeClassConditionalData(self):
        for j in range(len(self.trainingDataY)):
            label = self.trainingDataY[j][0]
            data  = self.trainingDataX[j]

            if(label not in self.classConditionalData):
                self.classConditionalData[label] = np.array(data)
            
            else:
                self.classConditionalData[label] = np.vstack((self.classConditionalData[label],data))
                    
        
    def computeMean(self, lst):
        n = len(lst)
        finalSum = None
        i = 0 
        for entry in lst:
            if(i == 0):
                finalSum = entry

            else:
                finalSum = finalSum+entry
            
            i = i+1
        
        finalArr = (1/n)*finalSum

        return finalArr
    
    def computeCovariance(self, lst):
        return np.cov(lst.T)

    
    def computeClassConditional(self):
        for label in self.classConditionalData:
            dataForLabel = self.classConditionalData[label]
            mean = self.computeMean(dataForLabel)
            covariance_matrix  = self.computeCovariance(dataForLabel)
            self.classConditionals[label] = (mean, covariance_matrix)

    
    def classify(self):
        for i in range (len(self.testingDataY)):
            data  = self.testingDataX[i]
            label = self.testingDataY[i][0]

            probabilityVals = []
            for j in range(0,10):
                (mean, covariance_matrix) = self.classConditionals[j]
                classConditional = multivariate_normal.pdf(data, mean, covariance_matrix, allow_singular=True)
                classPrior = self.classPriors[label]
                probability = classConditional*classPrior
                probabilityVals.append(probability)
            
            prediction  = np.argmax(probabilityVals)

            if(prediction == label):
                self.correctlyClassifed = self.correctlyClassifed + 1 
    
    def computeAccuracy(self):
        self.accuracy = self.correctlyClassifed/len(self.testingDataY)
        return self.accuracy

    def reset(self):
        self.rawData  =  data
        self.classPriors = dict()
        self.classConditionalData = dict()
        self.classConditionals  = dict()
        self.X = None
        self.Y = None
        self.correctlyClassifed = 0
        self.accuracy = None
        self.KDtree = None

        



class KNNClassifier():
    def __init__(self,k):
        self.rawData  =  data
        self.classPriors = dict()
        self.classConditionalData = dict()
        self.classConditionals  = dict()
        self.X = None
        self.Y = None
        self.correctlyClassifed = 0
        self.accuracy = None
        self.k = k

    def processData(self):
        self.X =  self.returnPrincipalComponents(self.rawData['X']).T
        self.Y = self.rawData['Y']  


    def returnPrincipalComponents(self, X_data):
        pca  = PCA(n_components=20)
        pca.fit(X_data.T)
        return pca.components_

    def splitTrainingTesting(self, split):
        entries =  len(self.X)
        trainingDataX  = self.X[0:int(entries*split)]
        testingDataX = self.X[int(entries*split):entries]

        trainingDataY  = self.Y[0:int(entries*split)]
        testingDataY = self.Y[int(entries*split):entries]

        self.trainingDataX =  trainingDataX
        self.trainingDataY =  trainingDataY 

        self.testingDataX =  testingDataX 
        self.testingDataY =  testingDataY 

        return (trainingDataX, trainingDataY), (testingDataX, testingDataY)

    def createKDtree(self):
        self.KDtree = KDTree(self.trainingDataX)

    def classify(self, norm):
        for i in range (len(self.testingDataY)):
            data  = self.testingDataX[i]
            label = self.testingDataY[i][0]

            d, i = self.KDtree.query(data,k=self.k,p=norm)

            if(self.k == 1):
                prediction = self.trainingDataY[i][0]
            
            else:
                predictions  = []
                for idx in i:
                    predictions.append(self.trainingDataY[idx][0])

                prediction = mode(predictions)[0]

            if(prediction == label):
                self.correctlyClassifed = self.correctlyClassifed + 1 
    
    def computeAccuracy(self):
        self.accuracy = self.correctlyClassifed/len(self.testingDataY)
        print(self.accuracy)
        return self.accuracy


    def reset(self):
        self.rawData  =  data
        self.classPriors = dict()
        self.classConditionalData = dict()
        self.classConditionals  = dict()
        self.X = None
        self.Y = None
        self.correctlyClassifed = 0
        self.accuracy = None
        self.KDtree = None

        



def runComparisonModels():
    classifier1 = mleClassifier()
    classifier2 = KNNClassifier(k=1)
    classifier1.processData()
    classifier2.processData()


    #line 165 singular matrix when input with 0.01
    trainingTestingSplit = [0.02,0.03,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    accuracyMLE = []

    accuracyKNN = []

    for trainingSplit in trainingTestingSplit:
        print("trainingSplit",trainingSplit)
        classifier1.processData()
        classifier1.splitTrainingTesting(trainingSplit)
        classifier1.computeClassPriors()
        classifier1.computeClassConditionalData()
        classifier1.computeClassConditional()
        classifier1.classify()
        acc1 = classifier1.computeAccuracy()
        print("Acc1",acc1)
        accuracyMLE.append(acc1)
        
        classifier1.reset()

        classifier2.processData()
        classifier2.splitTrainingTesting(trainingSplit)
        classifier2.createKDtree()
        classifier2.classify(norm=2)
        acc2 = classifier2.computeAccuracy()
        print("Acc2",acc2)

        accuracyKNN.append(acc2)
        classifier2.reset()



    plt.plot(trainingTestingSplit,accuracyMLE,"-b",label="MLE classifier accuracy")
    plt.plot(trainingTestingSplit,accuracyKNN,"-r",label="KNN classifier accuracy")
    plt.xlabel("Training Test Split (Proportion of data in the training set)")
    plt.ylabel("Accuracy")
    plt.ylim(0,1.0)
    plt.xlim(0,1.0)
    plt.legend()
    plt.show()

        


def testDifferentNorms():


    trainingTestingSplit = [0.0001,0.02,0.03,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    oneNormAcc = []
    twoNormAcc = []
    infNormAcc = []


    for trainingSplit in trainingTestingSplit:    
        classifier2 = KNNClassifier(k=1)
        classifier2.processData()
        classifier2.splitTrainingTesting(trainingSplit)
        classifier2.createKDtree()
        classifier2.classify(norm=1)
        accNorm1 = classifier2.computeAccuracy()
        oneNormAcc.append(accNorm1)
        classifier2.reset()

        classifier2.processData()
        classifier2.splitTrainingTesting(trainingSplit)
        classifier2.createKDtree()
        classifier2.classify(norm=2)
        accNorm2 = classifier2.computeAccuracy()
        twoNormAcc.append(accNorm2)
        classifier2.reset()

        classifier2.processData()
        classifier2.splitTrainingTesting(trainingSplit)
        classifier2.createKDtree()
        classifier2.classify(norm=np.inf)
        accNormInf = classifier2.computeAccuracy()
        infNormAcc.append(accNormInf)


        print("Acc Norm 1:",accNorm1)
        print("Acc Norm 2:",accNorm2)
        print("Acc Norm Inf:",accNormInf)

    plt.plot(trainingTestingSplit,oneNormAcc,"-b",label="1 norm accuracy")
    plt.plot(trainingTestingSplit,twoNormAcc,"-r",label="2 norm accuracy")
    plt.plot(trainingTestingSplit,infNormAcc,"-g",label="Inf norm accuracy")
    plt.xlabel("Training Test Split (Proportion of data in the training set)")
    plt.ylabel("Accuracy")
    plt.ylim(0.0,1.0)
    plt.xlim(0.0,1.0)
    plt.legend()
    plt.show()









        
def __main__():

    #testDifferentNorms()
    runComparisonModels()

    
    
    classifier1 = mleClassifier()
    classifier1.processData()
    classifier1.splitTrainingTesting(0.9)
    classifier1.computeClassPriors()
    classifier1.computeClassConditionalData()
    classifier1.computeClassConditional()
    classifier1.classify()
    classifier1.computeAccuracy()


    classifier2 = KNNClassifier(k=2)
    classifier2.processData()
    classifier2.splitTrainingTesting(0.9)
    classifier2.createKDtree()
    classifier2.classify(norm=2)
    classifier2.computeAccuracy()
    
    



if __name__ == "__main__":
    __main__()
