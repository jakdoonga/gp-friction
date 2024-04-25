import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gpflow
from scipy.cluster.vq import kmeans
import time

# The function of ground truth
def f(x_1, x_2):
    f_result = x_1 + np.exp(-0.5*x_2)
    return f_result

def obs(x_1, x_2):
    obs_result = f(x_1,x_2) + 0.1*np.random.normal(1)
    return obs_result

def GeneratePlotData(n_test):
    # Generate data to plot
    x_1 = np.linspace(-1.0, 1.0, n_test)
    x_2 = np.linspace(-1.0, 1.0, n_test)
    x_1_, x_2_ = np.meshgrid(x_1, x_2)
    f_ = f(x_1_, x_2_)
    return x_1_, x_2_, f_

def GenerateObsData(n_train):
    # Generate observation data
    x_1_obs = np.random.uniform(-2.0, 2.0, size=(n_train,1))
    x_2_obs = np.random.uniform(-2.0, 2.0, size=(n_train,1))
    x_1_obs_, x_2_obs_ = np.meshgrid(x_1_obs, x_2_obs)
    f_obs_ = obs(x_1_obs_, x_2_obs_)
    return x_1_obs_, x_2_obs_, f_obs_

def model_optimization(model):
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables)

def model_save(model):
    model.compiled_predict_f = tf.function(
        lambda xnew: model.predict_f(xnew, full_cov=False),
        input_signature =[tf.TensorSpec(shape=(None,2), dtype=tf.float64)],
        )

    save_dir="saved_model"
    tf.saved_model.save(model, save_dir)

if __name__ == '__main__':

    n_plot = 1000
    n_train = 100
    n_inducing = 50

    gpu_devices = tf.config.experimental.list_physical_devices('GPU')

    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    print(tf.__version__)

    x_1_plot, x_2_plot, f_plot = GeneratePlotData(n_plot)
    Xplot = np.vstack((x_1_plot.flatten(), x_2_plot.flatten())).T

    x_1_obs, x_2_obs, f_obs = GenerateObsData(n_train)
    X = np.vstack((x_1_obs.flatten(), x_2_obs.flatten())).T
    Y = f_obs.reshape(n_train*n_train,1)
    print(X.shape)
    print(Y.shape)

    # Subsample trainable variables
    inducing_variable, _ = kmeans(X,n_inducing)

    # Model creation and optimization
    model = gpflow.models.SGPR((X, Y), kernel=gpflow.kernels.RBF(),inducing_variable=inducing_variable)
    model_optimization(model)

    model_save(model)

    # model2 = gpflow.models.SGPR((X, Y), kernel=gpflow.kernels.RBF(),inducing_variable=inducing_variable)
    # params = gpflow.utilities.parameter_dict(model)
    # # print(params, X, Y, inducing_variable)
    # gpflow.utilities.multiple_assign(model2, params)
    # # np.savez('GPflow_2D_test.npz',params, X, Y, inducing_variable)

    # Model save
    # model_save(model)

    # f_mean, f_var = model.predict_f(Xplot, full_cov=False)
    # f_mean_ = f_mean.numpy()
    # f_mean_ = f_mean_.reshape(n_plot,n_plot)
    #
    # fig = plt.figure(1)
    # ax = fig.add_subplot(projection='3d')
    # ax.plot_surface(x_1_plot, x_2_plot, f_mean_)
    # ax.set_zlabel('f($x_1$,$x_2$) - SGPR')
    # plt.xlabel('$x_1$')
    # plt.ylabel('$x_2$')
    # plt.show()
    # # plt.savefig('SGP_result.png')
    #
    fig = plt.figure(1)
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(x_1_plot, x_2_plot, f_plot)
    ax.set_zlabel('f($x_1$,$x_2$) - ground truth')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.show()
    # plt.savefig('ground_truth.png')

