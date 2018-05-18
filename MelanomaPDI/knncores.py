#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 14:46:37 2018

@author: pesquisador2
"""
from sklearn.preprocessing import normalize
import cv2
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
#====================================
# 0 - Nevo
# 1 - Melanoma
#
# [0] = contraste;
# [1] = dissimilaridade;
# [2] = homogeneidade;
# [3] = energia;
# [4] = entropia;
#====================================

# Get data from the .csv files generated from the c++ program
trainData0 = genfromtxt('trainData.csv', delimiter=' ')
responses0 = genfromtxt('responses.csv', delimiter=' ')
testData0 = genfromtxt('testData.csv', delimiter=' ')
realData0 = genfromtxt('realData.csv', delimiter=' ')

# Convert to float32 data type
trainData0 = np.float32(trainData0)
responses0 = np.float32(responses0)
testData0 = np.float32(testData0)
realData0 = np.float32(realData0)

#Get data from .csv files generated from color features
cores_nevos = genfromtxt('nevos_cropped_cor.csv', delimiter=',')
cores_nevos = np.float32(cores_nevos)
cores_nevos = normalize(cores_nevos, axis=1, norm='max')

cores_melanomas = genfromtxt('melanomas_cropped_cor.csv', delimiter=',')
cores_melanomas = np.float32(cores_melanomas)
cores_melanomas = normalize(cores_melanomas, axis=1, norm='max')

#Setting the number of images to train and verify
test_number = genfromtxt('test_info.txt', delimiter=' ')
n_test_nevo0 = test_number[1]
n_test_melanoma0 = test_number[2]
n_test_nevo = 97
n_test_melanoma = 97
n_verif_nevo=100
n_verif_melanoma=100
linhascor,colcor=cores_melanomas.shape

responses=np.float32(np.ones(int(n_test_nevo)*2))
responses[0:np.int(n_test_nevo)]=0

realData=np.float32(np.ones(int(n_verif_nevo)*2))
realData[0:np.int(n_verif_nevo)]=0

trainData=np.zeros((np.int(n_test_nevo)*2,colcor))
trainData[0:n_test_nevo,:]=cores_nevos[0:n_test_nevo,:]
trainData[np.int(n_test_nevo):,:]=cores_melanomas[0:np.int(n_test_melanoma),:]

testData=np.zeros((np.int(n_verif_nevo)*2,colcor))
testData[0:n_verif_nevo,:]=cores_nevos[n_test_nevo:n_verif_nevo+n_test_nevo,:]
testData[np.int(n_verif_nevo):,:]=cores_melanomas[n_test_melanoma:np.int(n_test_melanoma)+n_verif_melanoma,:]

neib=9
acertos = 0
falsoPositivo = 0;
falsoNegativo = 0;
melanoma_Melanoma = 0;
nevo_Nevo = 0;



TrainDataSpecificos=np.zeros(responses0.size)
TestDataSpecificos=np.zeros(realData0.size)
TrainDataSpecificos=np.float32(TrainDataSpecificos).T
TestDataSpecificos=np.float32(TestDataSpecificos).T
select = [1,1,1,1,1]

#==============================================================================

for x in range(0,5):
    if select[x] == 1:
        TrainDataSpecificos = np.column_stack((TrainDataSpecificos,trainData0[:,x]))
        TestDataSpecificos = np.column_stack((TestDataSpecificos,testData0[:,x]))
TrainDataSpecificos = TrainDataSpecificos[:,1:]
TestDataSpecificos = TestDataSpecificos[:,1:]
#==============================================================================
        #KNN SPECIFIC FEATURES
#==============================================================================
knn3 = cv2.ml.KNearest_create()
knn3.train(TrainDataSpecificos,cv2.ml.ROW_SAMPLE,responses0)
ret, results, neighbours, dist = knn3.findNearest(TestDataSpecificos, neib)
for y in range(0, realData0.size):
    if realData0[y] == 1 and results[y] == 1:
       melanoma_Melanoma = melanoma_Melanoma + 1
    elif realData0[y] == 0 and results[y] == 0:
       nevo_Nevo = nevo_Nevo + 1
    elif realData0[y] == 0 and results[y] == 1:
      falsoPositivo = falsoPositivo + 1
    elif realData0[y] == 1 and results[y] == 0:
      falsoNegativo = falsoNegativo + 1   
success_rate = (melanoma_Melanoma + nevo_Nevo) / float(realData0.size)
error_rate = (falsoNegativo+falsoPositivo)/float(realData0.size)
acertos = 0

print ('Success Rate for specific features combined = {} - para {} imagens de treino e {} imagens de teste'.format(success_rate*100,trainData0.shape[0],testData0.shape[0]))
print()
print ('{}|{}'.format((melanoma_Melanoma * 100 / float(n_test_melanoma0)),(falsoNegativo * 100 / float(n_test_melanoma0))))
print ('{}|{}'.format((falsoPositivo * 100 / float(n_test_nevo0)),(nevo_Nevo * 100/float(n_test_nevo0))))
print()
print ('Falso Positivo (Nevo classificado como melanoma) ={} '.format(100*falsoPositivo/(falsoNegativo+falsoPositivo)))
print ('Falso Negativo (Melanoma classificado como nevo) = {}'.format(100*falsoNegativo/(falsoNegativo+falsoPositivo)))  
print ('-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx-')
print()
acertos = 0
falsoPositivo = 0;
falsoNegativo = 0;
melanoma_Melanoma = 0;
nevo_Nevo = 0;

#==============================================================================
        #KNN CORES
#==============================================================================

knn = cv2.ml.KNearest_create()
knn.train(np.float32(trainData),cv2.ml.ROW_SAMPLE,responses)
ret, results, neighbours, dist = knn.findNearest(np.float32(testData), neib)
for y in range(0, realData.size):
    if realData[y] == 1 and results[y] == 1:
       melanoma_Melanoma = melanoma_Melanoma + 1
    elif realData[y] == 0 and results[y] == 0:
       nevo_Nevo = nevo_Nevo + 1
    elif realData[y] == 0 and results[y] == 1:
      falsoPositivo = falsoPositivo + 1
    elif realData[y] == 1 and results[y] == 0:
      falsoNegativo = falsoNegativo + 1   
success_rate = (melanoma_Melanoma + nevo_Nevo) / float(realData.size)
error_rate = (falsoNegativo+falsoPositivo)/float(realData.size)
acertos = 0

print ('Success Rate for colors = {} - para {} imagens de treino e {} imagens de teste'.format(success_rate*100,n_test_nevo*2,n_verif_nevo*2))
print()
print ('{}|{}'.format((melanoma_Melanoma * 100 / float(n_verif_melanoma)),(falsoNegativo * 100 / float(n_verif_melanoma))))
print ('{}|{}'.format((falsoPositivo * 100 / float(n_verif_nevo)),(nevo_Nevo * 100/float(n_verif_nevo))))
print()
print ('Falso Positivo (Nevo classificado como melanoma) ={} '.format(100*falsoPositivo/(falsoNegativo+falsoPositivo)))
print ('Falso Negativo (Melanoma classificado como nevo) = {}'.format(100*falsoNegativo/(falsoNegativo+falsoPositivo)))
print ('-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx-')
print()   

