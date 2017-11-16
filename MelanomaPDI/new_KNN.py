
import cv2
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt


def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]

clear_all()

trainData = genfromtxt('trainData.csv', delimiter=' ')
responses = genfromtxt('responses.csv', delimiter=' ')
testData = genfromtxt('testData.csv', delimiter=' ')
realData = genfromtxt('realData.csv', delimiter=' ')

trainData = np.float32(trainData)
responses = np.float32(responses)
testData = np.float32(testData)
realData = np.float32(realData)

neib = 9;
contador = 0
for x in range(0, 5):
   knn = cv2.ml.KNearest_create()
   knn.train(np.float32([trainData[:,x]]).T,cv2.ml.ROW_SAMPLE,responses)
   ret, results, neighbours, dist = knn.findNearest(np.float32([testData[:,x]]).T, neib)
   if x == 0:
       bug = results
   for y in range(0,56):
       if realData[y] == results[y]:
           contador=contador+1
   success_rate = contador/float(56)
   print "Success Rate[",x,"] = ",success_rate*100,"%\n"
   contador = 0
   
knn2 = cv2.ml.KNearest_create()
knn2.train(trainData,cv2.ml.ROW_SAMPLE,responses)
ret, results, neighbours, dist = knn2.findNearest(testData, neib)
for y in range(0,56):
       if realData[y] == results[y]:
           contador=contador+1
success_rate = contador/float(56)
contador = 0
print "Success Rate for 5 features combined = ",success_rate*100,"%\n"

#=====================================================================
C=np.zeros(127)
D=np.zeros(56)
C=np.float32(C).T
D=np.float32(D).T
select = [0,0,0,1,0]

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
for y in range(0,56):
       if realData[y] == results[y]:
           contador=contador+1
success_rate = contador/float(56)
contador = 0
print "Success Rate for specific features combined = ",success_rate*100,"%\n"

#B = trainData[:,1]
#C = np.column_stack((A,B))
#B = np.float32([trainData[:,1]]).T
#C = np.float32([A,B]).T