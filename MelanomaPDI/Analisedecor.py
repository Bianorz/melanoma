#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 15:53:38 2018

@author: pesquisador2
"""

import numpy as np
#from scipy import ndimage
import cv2
import pandas as pd
from os import listdir
from os.path import isfile, join


mypath='/home/pesquisador/melanoma/MelanomaPDI/cropped_melanoma_database/'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
cores=(len(onlyfiles),9)
cores=np.zeros(cores)
for n in range(0, len(onlyfiles)):
    Itmp = cv2.imread( join(mypath,onlyfiles[n]) )
#Itmp=cv2.imread('/home/pesquisador2/melanoma/MelanomaPDI/melanoma_database/5.jpg')
    linhas,colunas,canais=Itmp.shape
    Itmphsv=cv2.cvtColor(Itmp,cv2.COLOR_BGR2HSV)

    print('Estamos na imagem {} melanoma'.format(n))
#f= open("data_color.txt","w+")
    for i in range(linhas):
        for j in range(colunas):
            if Itmphsv[i,j,0] >= 0 and Itmphsv[i,j,0] <=19:
                cores[n,0]=cores[n,0]+1
            if Itmphsv[i,j,0] >= 20 and Itmphsv[i,j,0] <=39:
                cores[n,1]=cores[n,1]+1
            if Itmphsv[i,j,0] >= 40 and Itmphsv[i,j,0] <=59:
                cores[n,2]=cores[n,2]+1
            if Itmphsv[i,j,0] >= 60 and Itmphsv[i,j,0] <=79:
                cores[n,3]=cores[n,3]+1
            if Itmphsv[i,j,0] >= 80 and Itmphsv[i,j,0] <=99:
                cores[n,4]=cores[n,4]+1
            if Itmphsv[i,j,0] >= 100 and Itmphsv[i,j,0] <=119:
                cores[n,5]=cores[n,5]+1
            if Itmphsv[i,j,0] >= 120 and Itmphsv[i,j,0] <=139:
                cores[n,6]=cores[n,6]+1
            if Itmphsv[i,j,0] >= 140 and Itmphsv[i,j,0] <=159:
                cores[n,7]=cores[n,7]+1
            if Itmphsv[i,j,0] >= 160 and Itmphsv[i,j,0] <=179:
                cores[n,8]=cores[n,8]+1

df = pd.DataFrame(cores)
df.to_csv("melanomas_cropped_cor.csv",header=None,index=None)
print(df)

mypath='/home/pesquisador/melanoma/MelanomaPDI/cropped_nevo_database/'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
cores=(len(onlyfiles),9)
cores=np.zeros(cores)
for n in range(0, len(onlyfiles)):
    Itmp = cv2.imread( join(mypath,onlyfiles[n]) )
#Itmp=cv2.imread('/home/pesquisador2/melanoma/MelanomaPDI/melanoma_database/5.jpg')
    linhas,colunas,canais=Itmp.shape
    Itmphsv=cv2.cvtColor(Itmp,cv2.COLOR_BGR2HSV)

    print('Estamos na imagem {} nevos'.format(n))
#f= open("data_color.txt","w+")
    for i in range(linhas):
        for j in range(colunas):
            if Itmphsv[i,j,0] >= 0 and Itmphsv[i,j,0] <=19:
                cores[n,0]=cores[n,0]+1
            if Itmphsv[i,j,0] >= 20 and Itmphsv[i,j,0] <=39:
                cores[n,1]=cores[n,1]+1
            if Itmphsv[i,j,0] >= 40 and Itmphsv[i,j,0] <=59:
                cores[n,2]=cores[n,2]+1
            if Itmphsv[i,j,0] >= 60 and Itmphsv[i,j,0] <=79:
                cores[n,3]=cores[n,3]+1
            if Itmphsv[i,j,0] >= 80 and Itmphsv[i,j,0] <=99:
                cores[n,4]=cores[n,4]+1
            if Itmphsv[i,j,0] >= 100 and Itmphsv[i,j,0] <=119:
                cores[n,5]=cores[n,5]+1
            if Itmphsv[i,j,0] >= 120 and Itmphsv[i,j,0] <=139:
                cores[n,6]=cores[n,6]+1
            if Itmphsv[i,j,0] >= 140 and Itmphsv[i,j,0] <=159:
                cores[n,7]=cores[n,7]+1
            if Itmphsv[i,j,0] >= 160 and Itmphsv[i,j,0] <=179:
                cores[n,8]=cores[n,8]+1

df = pd.DataFrame(cores)
df.to_csv("nevos_cropped_cor.csv",header=None,index=None)
print(df)
#f.write(cores)
#f.close
#totaldecores=np.sum(cores)
#totaldepixels=linhas*colunas
#print('O total de cores {} e o total de pixels{}'.format(totaldecores,totaldepixels))