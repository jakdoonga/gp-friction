import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import gpflow
from gpflow.ci_utils import reduce_in_tests
from gpflow.config import default_float

## sample data
## (1207,3) columns -> load 5/10/15
load1 = np.loadtxt('/home/zpqls/workspace/GP/fric_csv_raw/load1.csv', delimiter=',')
tau1 = np.loadtxt('/home/zpqls/workspace/GP/fric_csv_raw/tau1.csv', delimiter=',')

## choose load 15 
X = load1[:,2].reshape((load1.shape[0], 1))
Y = tau1[:,2].reshape((load1.shape[0], 1))

## create model
model = gpflow.models.GPR(
    (X, Y),
    kernel=gpflow.kernels.SquaredExponential(),
)

## train the model
opt = gpflow.optimizers.Scipy()
opt.minimize(model.training_loss, model.trainable_variables)

## wanna know f and Y might be at 0.5
Xnew = np.array([[35.0]])
model.predict_f(Xnew)
model.predict_y(Xnew)

## generate test pts for prediction
Xplot = np.linspace(0, 60, 1000)[:, None]
## predict f and Y
f_mean, f_var = model.predict_f(Xplot, full_cov=False)
y_mean, y_var = model.predict_y(Xplot)
## compute 95% confidence interval of f and Y
f_lower = f_mean - 1.96 * np.sqrt(f_var)
f_upper = f_mean + 1.96 * np.sqrt(f_var)
y_lower = y_mean - 1.96 * np.sqrt(y_var)
y_upper = y_mean + 1.96 * np.sqrt(y_var)

## plot result
plt.plot(X, Y, "kx", mew=2, label="input data")
plt.plot(Xplot, f_mean, "-", color="r", label="mean")
plt.plot(Xplot, f_lower, "--", color="C0", label="f 95% confidence")
plt.plot(Xplot, f_upper, "--", color="C0")
plt.fill_between(
    Xplot[:, 0], f_lower[:, 0], f_upper[:, 0], color="C0", alpha=0.1
)
plt.plot(Xplot, y_lower, ".", color="C0", label="Y 95% confidence")
plt.plot(Xplot, y_upper, ".", color="C0")
plt.fill_between(
    Xplot[:, 0], y_lower[:, 0], y_upper[:, 0], color="C0", alpha=0.1
)
plt.legend()
plt.show()