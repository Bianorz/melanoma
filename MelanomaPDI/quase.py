import cv2
import numpy as np
import matplotlib.pyplot as plt


# Feature set containing (x,y) values of 25 known/training data
#trainData = np.random.randint(0,100,(25,2)).astype(np.float32)

# Labels each one either Red or Blue with numbers 0 and 1
#responses = np.random.randint(0,2,(25,1)).astype(np.float32)

# Take Red families and plot them
#red = trainData[responses.ravel()==0]
import scipy.io
ocupadaDicio = scipy.io.loadmat('/home/bianor/ocupada.mat')
livreDicio = scipy.io.loadmat('/home/bianor/livre.mat')
trainDataDicio= scipy.io.loadmat('/home/bianor/trainData.mat')
responsesDicio= scipy.io.loadmat('/home/bianor/responses.mat')

ocupada = ocupadaDicio["ocupada"]
livre = livreDicio["livre"]
trainData = trainDataDicio["trainData"]
responses = responsesDicio["responses"]
newcomer = np.float_([[1000,0.93]])

ocupada = np.float32(ocupada)
livre = np.float32(livre)
trainData = np.float32(trainData)
responses = np.float32(responses)
newcomer = np.float32(newcomer)

plt.scatter(ocupada[:,0],ocupada[:,1],80,'r','^')
plt.scatter(livre[:,0],livre[:,1],80,'b','s')
axes = plt.gca()
axes.set_ylim([0,1])

plt.scatter(newcomer[:,0],newcomer[:,1],80,'g','o')
plt.show();

knn = cv2.ml.KNearest_create()
knn.train(trainData,cv2.ml.ROW_SAMPLE,responses)
ret, results, neighbours, dist = knn.findNearest(newcomer, 3)


print "result: ", results,"\n"
print "neighbours: ", neighbours,"\n"
print "distance: ", dist

plt.show()