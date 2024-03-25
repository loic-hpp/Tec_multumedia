import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from numpy import linalg as LA

images = []
images1 = []

dataPath = "/data/"
currDirectory = os.getcwd()
fileList = os.listdir(currDirectory + dataPath)

# Lectures des images et conversion RGB ->
for file in fileList:
    filePath = currDirectory + dataPath + file
    img = cv2.imread(filePath, cv2.IMREAD_COLOR)
    img2YUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)    # Conversion des images en YUV
    images.append(img2YUV)
    images1.append(img2YUV)
    
for image in images:
    
    moyY, moyU, moyV = np.mean(image, axis=(0,1))
    #print(f"{moyY} {moyU} {moyY}")
    
    # Calcul de la matrice de covariance des YUV
    covRGB = np.zeros((3,3), dtype = "double")
    for i in range(len(image)):
        for j in range(len(image[0])):
            vecTemp=[[image[i][j][0] - moyY], [image[i][j][1]] - moyU, [image[i][j][2] - moyV]]
            vecProdTemp = np.dot(vecTemp,np.transpose(vecTemp))
            covRGB = np.add(covRGB,vecProdTemp)

    covRGB = covRGB / image.size        
    #print(covRGB)
    # Calcul des vecteurs propres et valeurs propres
    eigval, eigvec = LA.eig(covRGB)
    #print(eigval)
    #print(eigvec)
    #print()
    eigvec = np.transpose(eigvec)
    eigvecsansAxe0 = np.copy(eigvec)
    eigvecsansAxe0[0,:] = [0.0,0.0,0.0]
    eigvecsansAxe1 = np.copy(eigvec)
    eigvecsansAxe1[1,:] = [0.0,0.0,0.0]
    eigvecsansAxe2 = np.copy(eigvec)
    eigvecsansAxe2[2,:] = [0.0,0.0,0.0]

    imageKLsansAxe0 = np.copy(image)
    imageKLsansAxe1 = np.copy(image)
    imageKLsansAxe2 = np.copy(image)

    vecMoy =[[moyY], [moyU], [moyV]]
    
    for image1 in images1:
        for i in range(len(image1)):
            for j in range(len(image1[0])):
                vecTemp = [[img[i][j][0]], [img[i][j][1]], [img[i][j][2]]]
                #a=Mb
                imageKLsansAxe0[i][j][:] = np.reshape(np.dot(eigvecsansAxe0,np.subtract(vecTemp,vecMoy)),(3))
                imageKLsansAxe1[i][j][:] = np.reshape(np.dot(eigvecsansAxe1,np.subtract(vecTemp,vecMoy)),(3))
                imageKLsansAxe2[i][j][:] = np.reshape(np.dot(eigvecsansAxe2,np.subtract(vecTemp,vecMoy)),(3)) 
    
    print(imageKLsansAxe0[10][10][:])