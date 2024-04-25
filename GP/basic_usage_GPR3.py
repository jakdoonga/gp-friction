import os
import numpy as np
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf

import gpflow
from gpflow.ci_utils import reduce_in_tests
from gpflow.config import default_float
from scipy.cluster.vq import kmeans

def getFname(directory):
    fname = os.listdir(directory)
    fname.sort()
    i = 0; fname1 = fname[i:i+3]+fname[i+9:i+12]+fname[i+18:i+21]+fname[i+27:i+30]
    i = 3; fname2 = fname[i:i+3]+fname[i+9:i+12]+fname[i+18:i+21]+fname[i+27:i+30]
    i = 6; fname3 = fname[i:i+3]+fname[i+9:i+12]+fname[i+18:i+21]+fname[i+27:i+30]
    fname = fname1+fname2+fname3

    return fname

def csv2array(fname, legNum):
    for i in range(0,len(fname)):
        if i//12 == legNum-1:
            if (i-i//12*12)//3 == 0:
                globals()['tau_down_{}_{}'.format(i//12+1, 5*(i%3+1))] = \
                    np.loadtxt('/home/zpqls/workspace/GP/fric_csv/{}'.format(fname[i]), delimiter=',')
            elif (i-i//12*12)//3 == 1:
                globals()['tau_up_{}_{}'.format(i//12+1, 5*(i%3+1))] = \
                    np.loadtxt('/home/zpqls/workspace/GP/fric_csv/{}'.format(fname[i]), delimiter=',')
            elif (i-i//12*12)//3 == 2:
                globals()['vel_down_{}_{}'.format(i//12+1, 5*(i%3+1))] = \
                    np.loadtxt('/home/zpqls/workspace/GP/fric_csv/{}'.format(fname[i]), delimiter=',')
            elif (i-i//12*12)//3 == 3:
                globals()['vel_up_{}_{}'.format(i//12+1, 5*(i%3+1))] = \
                    np.loadtxt('/home/zpqls/workspace/GP/fric_csv/{}'.format(fname[i]), delimiter=',')

def checkZero(array):
    flag = np.zeros(array.shape[1])
    for j in range(0,array.shape[1]):
        c = 0
        for i in range(0,array.shape[0]):
            if array[i,j] != 0:
                c += 1
        flag[j] = c
        if j ==4 or j == 8: flag[j] = 1025
    
    return flag

def extrTau(rawArray, xArray, type='mtr'):
    flag = checkZero(rawArray)
    for f in range(0,flag.shape[0]):
        if type == 'mtr':  ## for mtr
            if f//4 == 0 or f//4 == 2: 
                xArray = np.concatenate((xArray, rawArray[0:int(flag[f]),f]))
        else:
            if f//4 == 1 or f//4 == 3: ## for fric
                xArray = np.concatenate((xArray, rawArray[0:int(flag[f]),f]))
    
    return xArray

def extrVel(rawArray, xArray):
    flag = checkZero(rawArray)
    for f in range(0,flag.shape[0]):    
        xArray = np.concatenate((xArray, rawArray[0:int(flag[f]),f]))
    
    return xArray

def getX1():
    X1 = np.array([])
    X1 = extrVel(vel_up_1_5, X1)
    X1 = extrVel(- vel_down_1_5, X1)
    X1 = extrVel(vel_up_1_10, X1)
    X1 = extrVel(- vel_down_1_10, X1)
    X1 = extrVel(vel_up_1_15, X1)
    X1 = extrVel(- vel_down_1_15, X1)

    return X1
    
def getX2():
    X2 = np.array([])
    X2 = extrTau(tau_up_1_5, X2)
    X2 = extrTau(- tau_down_1_5, X2)
    X2 = extrTau(tau_up_1_10, X2)
    X2 = extrTau(- tau_down_1_10, X2)
    X2 = extrTau(tau_up_1_15, X2)
    X2 = extrTau(- tau_down_1_15, X2)

    return X2

def getY():
    Y = np.array([])
    Y = extrTau(tau_up_1_5, Y, 'fric')
    Y = extrTau(- tau_down_1_5, Y, 'fric')
    Y = extrTau(tau_up_1_10, Y, 'fric')
    Y = extrTau(- tau_down_1_10, Y, 'fric')
    Y = extrTau(tau_up_1_15, Y, 'fric')
    Y = extrTau(- tau_down_1_15, Y, 'fric')

    return Y

start = time.time()
## csv to np array  
fname = getFname('/home/zpqls/workspace/GP/fric_csv')
csv2array(fname, 1)

## X1 : rpm / X2 : tau mtr / Y : tau fric
X1 = getX1()
X2 = getX2()
X = np.concatenate((X1.reshape(-1,1), X2.reshape(-1,1)), axis=1)

Y = getY()
Y = Y.reshape(-1, 1)

# print(X.shape)
# print(Y.shape)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[:,0], X[:,1], Y, color='r', marker='o')
# ax.set_xlabel('rpm')
# ax.set_ylabel('mtr')
# ax.set_zlabel('fric')
# plt.show()

# ## shuffle and divide into train and test
# num_data = X.shape[0]
# indices = np.arange(num_data)
# np.random.shuffle(indices)
# X_shuffled = X[indices]
# Y_shuffled = Y[indices]

# train_size = int(0.3*num_data)
# X_train, X_test = X_shuffled[:train_size], X_shuffled[train_size:]
# Y_train, Y_test = Y_shuffled[:train_size], Y_shuffled[train_size:]


## create model
# model = gpflow.models.GPR(
#     data=(X, Y),
#     kernel=gpflow.kernels.SquaredExponential(),
# )
n_inducing = 25
inducing_variable, _ = kmeans(X,n_inducing)
model = gpflow.models.SGPR(data=(X, Y),
                           kernel=gpflow.kernels.SquaredExponential(
                                                 variance=0.5, lengthscales=[1.5, 3]),
                           inducing_variable=inducing_variable)

## train the model
opt = gpflow.optimizers.Scipy()
opt.minimize(model.training_loss, model.trainable_variables)

train_end = time.time()
print('\n\ntrain done : {}\n'.format(train_end-start))


## test the model
nn = 150
mm = 1
xx = np.random.uniform(min(X[:,0])*1/mm, max(X[:,0])*mm, (nn,1))
yy = np.random.uniform(min(X[:,1])*1/mm, max(X[:,1])*mm, (nn,1))
xx, yy = np.meshgrid(xx, yy)
X_test = np.vstack((xx.flatten(), yy.flatten())).T
f_mean, f_var = model.predict_f(X_test, full_cov=False)
y_mean, y_var = model.predict_y(X_test)

test_end = time.time()
print('\ntest done : {}\n'.format(test_end-train_end))

## compute 95% confidence interval of f and Y
f_lower = f_mean - 1.96 * np.sqrt(f_var)
f_upper = f_mean + 1.96 * np.sqrt(f_var)
y_lower = y_mean - 1.96 * np.sqrt(y_var)
y_upper = y_mean + 1.96 * np.sqrt(y_var)


## 예측 평균
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
ax1.scatter(X[:, 0], X[:, 1], Y, color='r', marker='o', label='Observations', alpha=0.2)
ax1.scatter(xx, yy, f_mean.numpy().reshape(xx.shape), alpha=0.3, color='C0')
# ax1.scatter(xx, yy, y_mean.numpy().reshape(xx.shape), alpha=0.3, color='g')
ax1.set_title('Mean')
ax1.set_xlabel(r'$\omega$ [rpm]')
ax1.set_ylabel(r'$\tau_{motor}$ [Nm]')
ax1.set_zlabel(r'$\tau_{fric}$ [Nm]')
plt.show()

# np.save('/home/zpqls/workspace/GP/npy/X', X)
# np.save('/home/zpqls/workspace/GP/npy/Y', Y)
# np.save('/home/zpqls/workspace/GP/npy/xx', xx)
# np.save('/home/zpqls/workspace/GP/npy/yy', yy)
# np.save('/home/zpqls/workspace/GP/npy/f_mean', f_mean)