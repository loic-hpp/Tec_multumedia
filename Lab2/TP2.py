# Importation des modules
import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from numpy import linalg as LA

# Lecture et affichage images
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
    #plt.imshow(img[:, :, ::-1])
    #plt.show()

# Obetention des 3 canaux et calcul de leur moyenne 
for file in fileList:
    filePath = currDirectory + dataPath + file
    img = cv.imread(filePath, cv.IMREAD_COLOR)
    moyB, moyG, moyR = np.mean(img, axis=(0,1))
    #b,g,r = cv.split(img)
    #moyB = np.mean(b)
    #moyG = np.mean(g)
    #moyR = np.mean(r)


# Calcul de la covariance et quantification
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
    #print(covRGB)
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

    imageKLsansAxe0 = np.copy(img)
    imageKLsansAxe1 = np.copy(img)
    imageKLsansAxe2 = np.copy(img)

    vecMoy =[[moyB], [moyG], [moyR]] 

    for i in range(len(img)):
        for j in range(len(img[0])):
            vecTemp = [[img[i][j][0]], [img[i][j][1]], [img[i][j][2]]]
            #a=Mb
            imageKLsansAxe0[i][j][:] = np.reshape(np.dot(eigvecsansAxe0,np.subtract(vecTemp,vecMoy)),(3))
            imageKLsansAxe1[i][j][:] = np.reshape(np.dot(eigvecsansAxe1,np.subtract(vecTemp,vecMoy)),(3))
            imageKLsansAxe2[i][j][:] = np.reshape(np.dot(eigvecsansAxe2,np.subtract(vecTemp,vecMoy)),(3))
    #print(imageKL)
    
    # Quantification de chaque niveau de l'image KL
    #b_KL, g_KL, r_KL = cv.split(imageKL)
    #a = b_KL.max() / 127
    #b = g_KL.max() / 127
    #c = r_KL.max() / 127
    #b_KL = np.floor(b_KL / a)
    #plt.imshow(b_KL, cmap="gray")
    #plt.show()

    # En faisant la transformée inverse, on peut voir les images qui résultent de la compression. b=inv(M)a. Dans le code, on utilse pinv \ 
    # (Pseudo-inverse), car la matrice est parfois singulière. Il faut faire b + moyenne.
    invEigvecsansAxe0 = LA.pinv(eigvecsansAxe0);
    invEigvecsansAxe1 = LA.pinv(eigvecsansAxe1);
    invEigvecsansAxe2 = LA.pinv(eigvecsansAxe2);

     
    imageRGBsansAxe0 = np.copy(img)
    imageRGBsansAxe1 = np.copy(img)
    imageRGBsansAxe2 = np.copy(img)

    for i in range(len(img)):
        for j in range(len(img[0])):
        #b=inv(M)a
            vecTempsansAxe0=[[imageKLsansAxe0[i][j][0]], [imageKLsansAxe0[i][j][1]], [imageKLsansAxe0[i][j][2]]]
            vecTempsansAxe1=[[imageKLsansAxe1[i][j][0]], [imageKLsansAxe1[i][j][1]], [imageKLsansAxe1[i][j][2]]]
            vecTempsansAxe2=[[imageKLsansAxe2[i][j][0]], [imageKLsansAxe2[i][j][1]], [imageKLsansAxe2[i][j][2]]]     
            imageRGBsansAxe0[i][j][:] = np.add(np.reshape(np.dot(invEigvecsansAxe0,vecTempsansAxe0),(3)),vecMoy)
            imageRGBsansAxe1[i][j][:] = np.add(np.reshape(np.dot(invEigvecsansAxe1,vecTempsansAxe1),(3)),vecMoy)
            imageRGBsansAxe2[i][j][:] = np.add(np.reshape(np.dot(invEigvecsansAxe2,vecTempsansAxe2),(3)),vecMoy)
    fig2 = py.figure(figsize = (10,10))
    imageout = np.clip(imageRGBsansAxe0,0,255)
    imageout= imageout.astype('uint8')
    plt.imshow(imageout)
    plt.show() 

    

    
