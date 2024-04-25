import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import gpflow
from gpflow.ci_utils import reduce_in_tests
from gpflow.config import default_float

## sample data
X = np.array(
    [
        [0.865], [0.666], [0.804], [0.771], [0.147], [0.866], [0.007], [0.026],
        [0.171], [0.889], [0.243], [0.028],
    ]
)
Y = np.array(
    [
        [1.57], [3.48], [3.12], [3.91], [3.07], [1.35], [3.80], [3.82], [3.49],
        [1.30], [4.00], [3.82],
    ]
)

# ## sample data - nosiy sinx
# num_train_data = reduce_in_tests(100)
# num_test_data = reduce_in_tests(500)

# X = tf.random.uniform((num_train_data, 1), dtype=default_float()) * 10
# Xtest = tf.random.uniform((num_test_data, 1), dtype=default_float()) * 10

# def noisy_sin(x):
#     return tf.math.sin(x) + 0.1 * tf.random.normal(
#         x.shape, dtype=default_float()
#     )
# Y = noisy_sin(X)
# Ytest = noisy_sin(Xtest)


# plt.plot(X, Y, "kx", mew=2)
# plt.show()

## create model
model = gpflow.models.GPR(
    (X, Y),
    kernel=gpflow.kernels.SquaredExponential(),
)

## train the model
opt = gpflow.optimizers.Scipy()
opt.minimize(model.training_loss, model.trainable_variables)

## wanna know f and Y might be at 0.5
Xnew = np.array([[0.5]])
model.predict_f(Xnew)
model.predict_y(Xnew)

## generate test pts for prediction
Xplot = np.linspace(-0.1, 1.1, 100)[:, None]
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
plt.plot(Xplot, f_mean, "-", color="C0", label="mean")
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