
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
trainData = genfromtxt('trainData.csv', delimiter=' ')
responses = genfromtxt('responses.csv', delimiter=' ')
testData = genfromtxt('testData.csv', delimiter=' ')
realData = genfromtxt('realData.csv', delimiter=' ')

# Convert to float32 data type
trainData = np.float32(trainData)
responses = np.float32(responses)
testData = np.float32(testData)
realData = np.float32(realData)

# Get the number of nevos and melanomas for test
test_number = genfromtxt('test_info.txt', delimiter=' ')
n_test_nevo = test_number[1]
n_test_melanoma = test_number[2]

# Number of neighbors used in the KNN
neib = 9;

# Inicio da  do erro
acertos = 0

# Matriz de confusao
falsoPositivo = 0; #  nevo e o algoritmo indicou melanoma
falsoNegativo = 0; #  melanoma e o algoritmo indicou nevo
melanoma_Melanoma = 0; #  melanoma e o algoritmo indicou melanoma
nevo_Nevo = 0;  #  nevo e o algoritmo indicou nevo
x = 3;
#print "class = ", x, "\n"  # Imprime a classe a ser analisada, 3 para energia
knn = cv2.ml.KNearest_create()  # Crio o classificador KNN 
knn.train(np.float32([trainData[:, x]]).T, cv2.ml.ROW_SAMPLE, responses)  # Realizo o treinamento utilizando
# apenas a classe 3
ret, results, neighbours, dist = knn.findNearest(np.float32([testData[:, x]]).T, neib)
for y in range(0, realData.size):
   if realData[y] == 1 and results[y] == 1:
       melanoma_Melanoma = melanoma_Melanoma + 1
   elif realData[y] == 0 and results[y] == 0:
       nevo_Nevo = nevo_Nevo + 1
   elif realData[y] == 0 and results[y] == 1:
      falsoPositivo = falsoPositivo + 1
   else:
      falsoNegativo = falsoNegativo + 1   
success_rate = (melanoma_Melanoma + nevo_Nevo) / float(realData.size)
print "Sucess Rate[", x, "] = ", success_rate * 100, "%\n"
print "", melanoma_Melanoma * 100 / float(n_test_melanoma), "|", falsoNegativo * 100 / float(n_test_melanoma), "\n"
print "", falsoPositivo * 100 / float(n_test_nevo), "|", nevo_Nevo * 100 / float(n_test_nevo), "\n"