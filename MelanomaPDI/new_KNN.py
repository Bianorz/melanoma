
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


acertos = 0
falsoPositivo = 0;
falsoNegativo = 0;
melanoma_Melanoma = 0;
nevo_Nevo = 0;
for x in range(0, 5):
   print "class = ",x,"\n" 
   knn = cv2.ml.KNearest_create()
   knn.train(np.float32([trainData[:,x]]).T,cv2.ml.ROW_SAMPLE,responses)
   ret, results, neighbours, dist = knn.findNearest(np.float32([testData[:,x]]).T, neib)
   if x == 0:
       bug = results
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
   print "-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx-"
   #print "Falso Positivo[",x,"] = ",100*falsoPositivo/(falsoNegativo+falsoPositivo),"%\n"
   #print "Falso Negativo[",x,"] = ",100*falsoNegativo/(falsoNegativo+falsoPositivo),"%\n\n"   
      
   acertos = 0
   falsoNegativo = 0
   falsoPositivo = 0
   melanoma_Melanoma=0;
   nevo_Nevo = 0;
   
   
knn2 = cv2.ml.KNearest_create()
knn2.train(trainData,cv2.ml.ROW_SAMPLE,responses)
ret, results, neighbours, dist = knn2.findNearest(testData, neib)
for y in range(0,realData.size):
       if realData[y] == results[y]:
           acertos=acertos+1
       elif realData[y] == 0 and results[y] == 1:
          falsoPositivo=falsoPositivo+1
       else:
          falsoNegativo=falsoNegativo+1
success_rate = acertos/float(realData.size)
error_rate = (falsoNegativo+falsoPositivo)/float(realData.size)
acertos = 0
print "Error Rate for 5 features combined = ",100-success_rate*100,"%\n"
print "Falso Positivo = ",100*falsoPositivo/(falsoNegativo+falsoPositivo),"%\n"
print "Falso Negativo = ",100*falsoNegativo/(falsoNegativo+falsoPositivo),"%\n\n"   
acertos = 0
falsoNegativo = 0
falsoPositivo = 0

#=====================================================================
C=np.zeros(responses.size)
D=np.zeros(realData.size)
C=np.float32(C).T
D=np.float32(D).T
select = [0,1,1,1,0]

#==============================================================================
for x in range(0,5):
    if select[x] == 1:
        C = np.column_stack((C,trainData[:,x]))
        D = np.column_stack((D,testData[:,x]))
#==============================================================================
#==============================================================================
C = C[:,1:]
D = D[:,1:]
knn3 = cv2.ml.KNearest_create()
knn3.train(C,cv2.ml.ROW_SAMPLE,responses)
ret, results, neighbours, dist = knn3.findNearest(D, neib)
for y in range(0,realData.size):
       if realData[y] == results[y]:
           acertos=acertos+1
       elif realData[y] == 0 and results[y] == 1:
          falsoPositivo=falsoPositivo+1
       else:
          falsoNegativo=falsoNegativo+1
success_rate = acertos/float(realData.size)
error_rate = (falsoNegativo+falsoPositivo)/float(realData.size)
acertos = 0
print "Error Rate for specific features combined = ",100-success_rate*100,"%\n"
print "Falso Positivo = ",100*falsoPositivo/(falsoNegativo+falsoPositivo),"%\n"
print "Falso Negativo = ",100*falsoNegativo/(falsoNegativo+falsoPositivo),"%\n\n"   
acertos = 0
falsoNegativo = 0
falsoPositivo = 0
