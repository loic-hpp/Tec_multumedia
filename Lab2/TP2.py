# Importation des modules
import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from numpy import linalg as LA

# Lecture des fichiers images
dataPath = "/data/"
currDirectory = os.getcwd()
fileList = os.listdir(currDirectory + dataPath)

for file in fileList:
    filePath = currDirectory + dataPath + file
    img = cv.imread(filePath, cv.IMREAD_COLOR)
    b,g,r = cv.split(img)
    img2YUV = cv.cvtColor(img, cv.COLOR_BGR2YUV) # Conversion des images en YUV
    v,u,y = cv.split(img2YUV)
    if img is None:
        quit("Pas d'image...")
    plt.imshow(img)
    
plt.show

# Obetention des 3 canaux et calcul de leur moyenne 
for file in fileList:
    filePath = currDirectory + dataPath + file
    img = cv.imread(filePath, cv.IMREAD_COLOR)
    moyB, moyG, moyR = np.mean(img, axis=(0,1))
    #b,g,r = cv.split(img)
    #moyB = np.mean(b)
    #moyG = np.mean(g)
    #moyR = np.mean(r)

# Calcul de la covariance
covRGB = np.zeros((3,3), dtype = "double")
for file in fileList:
    filePath = currDirectory + dataPath + file
    img = cv.imread(filePath, cv.IMREAD_COLOR)
    for i in range(len(img)):
        for j in range(len(img[0])):
            vecTemp=[[img[i][j][0] - moyB], [img[i][j][1]] - moyG, [img[i][j][2] - moyR]]
            vecProdTemp = np.dot(vecTemp,np.transpose(vecTemp))
            covRGB = np.add(covRGB,vecProdTemp)
    covRGB = covRGB / img.size       
    print(covRGB)
    eigval, eigvec = LA.eig(covRGB)
    print(eigval)
    print(eigvec)
    print()
    
