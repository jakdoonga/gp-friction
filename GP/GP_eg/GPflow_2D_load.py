from typing import Callable, Tuple
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gpflow
import time

def GeneratePlotData(n_test):
    # Generate data to plot
    x_1 = np.linspace(-1.0, 1.0, n_test)
    x_2 = np.linspace(-1.0, 1.0, n_test)
    x_1_, x_2_ = np.meshgrid(x_1, x_2)
    return x_1_, x_2_

if __name__ == "__main__":
    n_plot = 100

    save_dir = "saved_model"
    loaded_model = tf.saved_model.load(save_dir)
    print(loaded_model.compiled_predict_f)
    print(hasattr(loaded_model, 'compiled_predict_f'))
    # print(params)

    x_1_plot, x_2_plot = GeneratePlotData(n_plot)

    Xplot = np.vstack((x_1_plot.flatten(), x_2_plot.flatten())).T

    f_mean, f_var = loaded_model.compiled_predict_f(Xplot)
    f_mean_ = f_mean.numpy()
    f_mean_ = f_mean_.reshape(n_plot,n_plot)

    fig = plt.figure(1)
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(x_1_plot, x_2_plot, f_mean_)
    ax.set_zlabel('f($x_1$,$x_2$) - SGPR')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.show()