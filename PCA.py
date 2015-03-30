import numpy as np
from scipy import stats as s
from scipy.stats import multivariate_normal as mn
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

H = np.loadtxt('/home/user/PRTakeHome1/train_sp2015_v14')
St= np.loadtxt('/home/user/PRTakeHome1/test_sp2015_v14')

X = H

mean_vec = np.mean(X, axis=0)
#cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
#print('Covariance matrix \n%s' %cov_mat)

X_SubMean = np.zeros((15000, 4))
for i in range(0, 15000):
    X_SubMean[i] = X[i] - mean_vec
print X_SubMean

Sx_SubMean = np.zeros((15000, 4))
for i in range(0, 15000):
    Sx_SubMean[i] = St[i] - mean_vec
print Sx_SubMean

cv_mt = np.cov(X_SubMean.T)
print cv_mt

eig_vals, eig_vecs = np.linalg.eig(cv_mt)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

EVecsToBeConsiderd = eig_vecs[0:4, 0:2].T

New = EVecsToBeConsiderd.dot(  X_SubMean.T )
print New.shape

SNew = EVecsToBeConsiderd.dot(  Sx_SubMean.T )
print SNew.shape

data = New.T
Sdata = SNew.T

classOneMeanVector = np.zeros((2,1))
for j in range(0,2):
    sum = 0
    for i in range(0,5000):
        sum = sum + data[i][j]
    classOneMeanVector[j][0] = sum / 5000
print classOneMeanVector.transpose()
print classOneMeanVector.T
classOneTrainingData = data[0:5000,:]
summation = np.zeros((2,1))
for i in range(0, 5000):
    temp = classOneTrainingData[i] - classOneMeanVector.T[0]
    summation = summation + np.outer(temp.transpose(), temp.transpose())
estimatedCovMatrixClassOne = summation/4999
print estimatedCovMatrixClassOne
varOne = mn( mean = classOneMeanVector.T[0], cov = estimatedCovMatrixClassOne)

classTwoMeanVector = np.zeros((2,1))
for j in range(0,2):
    sum = 0
    for i in range(5000,10000):
        sum = sum + data[i][j]
    classTwoMeanVector[j][0] = sum / 5000
print classTwoMeanVector.transpose()
#print classOneMeanVector.T
classTwoTrainingData = data[5000:10000,:]
summation = np.zeros((2,1))
for i in range(0, 5000):
    temp = classTwoTrainingData[i] - classTwoMeanVector.T[0]
    summation = summation + np.outer(temp.transpose(), temp.transpose())
estimatedCovMatrixClassTwo = summation/4999
print estimatedCovMatrixClassTwo
varTwo = mn( mean = classTwoMeanVector.T[0], cov = estimatedCovMatrixClassTwo)

classThreeMeanVector = np.zeros((2,1))
for j in range(0,2):
    sum = 0
    for i in range(10000,15000):
        sum = sum + data[i][j]
    classThreeMeanVector[j][0] = sum / 5000
print classThreeMeanVector.transpose()
#print classOneMeanVector.T
classThreeTrainingData = data[10000:15000,:]
summation = np.zeros((2,1))
for i in range(0, 5000):
    temp = classThreeTrainingData[i] - classThreeMeanVector.T[0]
    summation = summation + np.outer(temp.transpose(), temp.transpose())
estimatedCovMatrixClassThree = summation/4999
print estimatedCovMatrixClassThree
varThree = mn( mean = classThreeMeanVector.T[0], cov = estimatedCovMatrixClassThree)

def findMax(a, b, c):
    if a > b and a > c:
        return "1"
    elif b > a and b > c:
        return "2"
    else:
        return "3"

f = open('/home/user/Desktop/TrainingRealityCheckTake2','w')
for i in range(0, 15000):
    f.write( findMax(  varOne.pdf( data[i:i+1,:][0] ), varTwo.pdf( data[i:i+1,:][0] ), varThree.pdf( data[i:i+1,:][0] ) )+'\n')
f.close()

def findMaxInReturn(a, b, c):
    if a > b and a > c:
        return 1
    elif b > a and b > c:
        return 2
    else:
        return 3

w11 = 0
w12 = 0
w13 = 0
for i in range(0, 5000):
    a = findMaxInReturn(  varOne.pdf( data[i:i+1,:][0] ), varTwo.pdf( data[i:i+1,:][0] ), varThree.pdf( data[i:i+1,:][0] ) )
    if a == 1:
        w11 = w11 + 1
    elif a == 2:
        w12 = w12 + 1
    else:
        w13 = w13 + 1
print w11
print w12
print w13

w21 = 0
w22 = 0
w23 = 0
for i in range(5000, 10000):
    a = findMaxInReturn(  varOne.pdf( data[i:i+1,:][0] ), varTwo.pdf( data[i:i+1,:][0] ), varThree.pdf( data[i:i+1,:][0] ) )
    if a == 1:
        w21 = w21 + 1
    elif a == 2:
        w22 = w22 + 1
    else:
        w23 = w23 + 1
print w21
print w22
print w23

w31 = 0
w32 = 0
w33 = 0
for i in range(10000, 15000):
    a = findMaxInReturn(  varOne.pdf( data[i:i+1,:][0] ), varTwo.pdf( data[i:i+1,:][0] ), varThree.pdf( data[i:i+1,:][0] ) )
    if a == 1:
        w31 = w31 + 1
    elif a == 2:
        w32 = w32 + 1
    else:
        w33 = w33 + 1
print w31
print w32
print w33

arr = [2,3,1,3,1,2]
err = 0
cof_mat = np.zeros((3,3))
for i in range(15000):
    retrievedRes = findMaxInReturn(  varOne.pdf( Sdata[i:i+1,:][0] ), varTwo.pdf( Sdata[i:i+1,:][0] ), varThree.pdf( Sdata[i:i+1,:][0] ) )
    if(  retrievedRes != arr[i%6]):
        err = err + 1
    cof_mat[arr[i%6]-1][retrievedRes-1] = cof_mat[arr[i%6]-1][retrievedRes-1] + 1
print err
print cof_mat


