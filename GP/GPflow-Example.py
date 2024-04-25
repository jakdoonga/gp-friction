# %% [markdown]
# <!--NOTEBOOK_HEADER-->
# *This notebook contains material from [cbe67701-uncertainty-quantification](https://ndcbe.github.io/cbe67701-uncertainty-quantification);
# content is available [on Github](https://github.com/ndcbe/cbe67701-uncertainty-quantification.git).*

# %% [markdown]
# <!--NAVIGATION-->
# < [10.0 Gaussian Process Emulators and Surrogate Models](https://ndcbe.github.io/cbe67701-uncertainty-quantification/10.00-Gaussian-Process-Emulators-and-Surrogate-Models.html) | [Contents](toc.html) | [10.2 A simple example of Bayesian quadrature](https://ndcbe.github.io/cbe67701-uncertainty-quantification/10.02-Bayesian-quadrature.html)<p><a href="https://colab.research.google.com/github/ndcbe/cbe67701-uncertainty-quantification/blob/master/docs/10.01-Contributed-Example.ipynb"> <img align="left" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" title="Open in Google Colaboratory"></a><p><a href="https://ndcbe.github.io/cbe67701-uncertainty-quantification/10.01-Contributed-Example.ipynb"> <img align="left" src="https://img.shields.io/badge/Github-Download-blue.svg" alt="Download" title="Download Notebook"></a>

# %% [markdown]
# # 10.1 Using GPflow package for Gaussian Process Regression
# 
# 
# Created by Bridgette Befort (bbefort@nd.edu)
# 
# The following example was adapted from:
# 
# De G. Matthews, A. G., Van Der Wilk, M., Nickson, T., Fujii, K., Boukouvalas, A., León-Villagrá, P., ... & Hensman, J. (2017). GPflow: A Gaussian process library using TensorFlow. The Journal of Machine Learning Research, 18(1), 1299-1304.
# 
# McClarren, Ryan G (2018). Uncertainty Quantification and Predictive Computational Science: A Foundation for Physical Scientists and Engineers, Chapter 10: Gaussian Process Emulators and Surrogate Models, Springer, https://link.springer.com/chapter/10.1007/978-3-319-99525-0_10

# %% [markdown]
# ## 10.1.1 Objectives and Organization
# 
# 1. GPflow example for the function $y = sin(x) + cos(x)$
#   * Setup steps
#   * Kernels
#   * Varying amount of train/test data
#     
#     
# 2. Apply GPflow tool to shock breakout time dataset
#   * Without scaling
#   * With scaling
#   

# %% [markdown]
# ## 10.1.2 Import Libraries
# 
# Note: GPflow needs to be installed
# 
# https://gpflow.readthedocs.io/en/master/intro.html
# 
# ACTION ITEM: Streamline this installation on Colab. Which version of GPFlow should be installed?

# %%
import numpy as np
import pandas as pd
import unyt as u
import matplotlib.pyplot as plt
import gpflow
import tensorflow as tf
from gpflow.utilities import print_summary

# %% [markdown]
# ## 10.1.3 Define Functions

# %%
def shuffle_and_split(df, n_params, fraction_train=0.8):
    """Randomly shuffles the DataFrame and extracts the train and test sets
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe with the
    n_params : int
        Number of parameters in the model
    fraction_train : float
        Fraction to use as training data. The remainder will be used for testing. Default is 0.8
    Returns
    -------
    x_train : np.ndarray
        Training inputs
    y_train : np.ndarray
        Training results
    x_test : np.ndarray
        Testing inputs
    y_test : np.ndarray
        Testing results
    """

    # Return values for all samples (liquid and vapor)
    data = df.values
    fraction_test = 1.0 - fraction_train
    total_entries = data.shape[0]
    train_entries = int(total_entries * fraction_train)
    # Shuffle the data before splitting train/test sets
    np.random.shuffle(data)

    # x = params, y = output
    x_train = data[:train_entries, : n_params].astype(np.float64)
    y_train = data[:train_entries, -1].astype(np.float64)
    x_test = data[train_entries:, : n_params].astype(np.float64)
    y_test = data[train_entries:, -1].astype(np.float64)

    return x_train, y_train, x_test, y_test

# %%
def run_gpflow_scipy(x_train, y_train, kernel):
    """Fits GP model to the training data
    Parameters
    ----------
    x_train : np.ndarray
        Training inputs
    y_train : np.ndarray
        Training results
    kernel : function
        GP flow kernel function
    Returns
    -------
    model : 
        fitted GP flow model
    """
    # Create the model
    model = gpflow.models.GPR(
        data=(x_train, y_train.reshape(-1, 1)),
        kernel=kernel,
        mean_function=gpflow.mean_functions.Linear(
            A=np.zeros(x_train.shape[1]).reshape(-1, 1)
        ),
    )

    # Print initial values
    print_summary(model, fmt="notebook")

    # Optimize model with scipy
    optimizer = gpflow.optimizers.Scipy()
    optimizer.minimize(model.training_loss, model.trainable_variables)

    # Print the optimized values
    print_summary(model, fmt="notebook")

    # Return the model
    return model

# %%
def plot_models(models, x_data, y_data, xylim_low=0, xylim_high=1):
    """Plot the performance of one or more GP models for some data x_data
    Parameters
    ----------
    models : dict { label : model }
        Each model to be plotted (value, GPFlow model) is provided
        with a label (key, string)
    x_data : np.array
        data to create model predictions for
    y_data : np.ndarray
        correct answer
    xylim_low : float, opt
        lower x and y limits of the plot, default 0
    xylim_high : float, opt
        upper x and y limits of the plot, default 1
    Returns
    -------
    """

    plt.plot(
        np.arange(xylim_low, xylim_high + 100, 100),
        np.arange(xylim_low, xylim_high + 100, 100),
        color="xkcd:blue grey",
        label="y=x",
    )

    for (label, model) in models.items():
        gp_mu, gp_var = model.predict_f(x_data)
        y_data_physical = y_data
        gp_mu_physical = gp_mu
        plt.scatter(y_data_physical, gp_mu_physical, label=label)
        sumsqerr = np.sum((gp_mu_physical - y_data_physical.reshape(-1, 1)) ** 2)
        print("Model: {}. Sum squared err: {:f}".format(label, sumsqerr))

    plt.xlim(xylim_low, xylim_high)
    plt.ylim(xylim_low, xylim_high)
    plt.xlabel("Actual")
    plt.ylabel("Model Prediction")
    plt.legend()
    ax = plt.gca()
    ax.set_aspect("equal", "box")

# %% [markdown]
# ## 10.1.4 GPFlow Example

# %% [markdown]
# **Objective**: Use GP flow to predict output of $y = sin(x) + cos(x)$

# %% [markdown]
# ### 10.1.4.1 Setup

# %% [markdown]
# #### 10.1.4.1.1 Step 1: Generate dataset

# %%
#Specify number of samples
n = 25

#Generate samples of x
x = np.random.rand(n,1)

#Calculate y
y = np.sin(x) + np.cos(x)

#Visualize
plt.plot(x,y,'.')
plt.xlabel('x')
plt.ylabel('y')

# %% [markdown]
# #### 10.1.4.1.2 Step 2: Split into train/test sets

# %%
#To use shuffle_and_split function, the data needs to be in a pandas dataframe
data = np.concatenate((x,y),axis=1)
data = pd.DataFrame(data,columns=['x','y'])

# %%
#Specify number of params
n_params = 1

#Apply shuffle_and_split function
x_train, y_train, x_test, y_test = shuffle_and_split(data, n_params, fraction_train=0.8)

# %% [markdown]
# #### 10.1.4.1.3 Step 3: Fit GP model

# %%
# Fit model--using RBF kernel
model_RBF = run_gpflow_scipy(x_train, y_train, gpflow.kernels.RBF(lengthscales=np.ones(n_params)))
model = {'RBF': model_RBF}

# %% [markdown]
# #### 10.1.4.1.4 Step 4: Compare Models

# %% [markdown]
# ##### 10.1.4.1.4.1 Train

# %%
plot_models(model, x_train, y_train, 0, 2)

# %% [markdown]
# ##### 10.1.4.1.4.2 Test

# %%
plot_models(model, x_test, y_test, 0, 2)

# %% [markdown]
# #### 10.1.4.1.5 Step 5: Analyze model predictions 

# %%
def plot_function(models,train,test):
    
    """Plot the performance (mean and variance) of one or more GP models along with the original train and test data
    Parameters
    ----------
    models : dict { label : model }
        Each model to be plotted (value, GPFlow model) is provided
        with a label (key, string)
    train : np.array
        array of training data, both and x and y values
    test : np.ndarray
        array of test data, both and x and y values
    Returns
    -------
    """
    #x data samples
    xx = np.linspace(0, 1.0, 100)[:,None]
    
    for (label, model) in models.items():
        
        #use model to predict output (y) given x data
        mean, var = model.predict_f(xx)
        #plot mean as line
        plt.plot(xx, mean, lw=2, label="GP model" + label)
        #plot variance as a shaded area
        plt.fill_between(
            xx[:, 0],
            mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
            mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
            alpha=0.25,
        )

    #Plot training and testing points
    if train.shape[0] > 0:
        x_train = train[:, 0]
        y_train = train[:, 1]
        plt.plot(x_train, y_train, "s", color="black", label="Train")
    if test.shape[0] > 0:
        x_test = test[:, 0]
        y_test = test[:, 1]
        plt.plot(x_test, y_test, "ro", label="Test")
        
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

# %%
#Make sure the y training and testing data has the correct shape
y_train.shape = (20,1)
y_test.shape = (5,1)
#Make arrays
train = np.concatenate((x_train,y_train),axis=1)
test = np.concatenate((x_test,y_test),axis=1)
 
plot_function(model,train,test)

# %% [markdown]
# ### 10.1.4.2 What happens if we use different kernels?

# %% [markdown]
# GP flow has many different available kernel functions: https://gpflow.readthedocs.io/en/master/gpflow/kernels/
# 
# Here we will examine a few options:
# 
# 1. Constant
# 
# $k(x,y) = \sigma^2$
# 
# where $\sigma^2$ is the variance parameter
# 
# 2. Linear
# 
# $k(x,y) = (\sigma^2xy+\gamma)^d$
# 
# where $\sigma^2$ is the variance parameter, $\gamma$ is the offset parameter, and $d$ is the degree parameter
# 
# 3. Radial Basis Function
# 
# $k(r) = \sigma^2 exp[-\frac{r^2}{2}]$
# 
# where $r$ is the Euclidean distance and $\sigma^2$ is the variance parameter
# 
# 4. Cosine
# 
# $k(r) = \sigma^2cos(2\pi d)$
# 
# where $r$ is the Euclidean distance, $\sigma^2$ is the variance parameter, and $d$ is the sum of the per-dimension differences between the input points scaled by the lenghtscale parameter $l$ 
# 
# 5. Matern12
# 
# $k(r) = \sigma^2exp[-r]$
# 
# where $r$ is the Euclidean distance and $\sigma^2$ is the variance parameter
# 
# 6. Matern32
# 
# $k(r) = \sigma^2(1+\sqrt{3}r)exp[-\sqrt{3}r]$
# 
# where $r$ is the Euclidean distance and $\sigma^2$ is the variance parameter
# 
# 6. Matern52
# 
# $k(r) = \sigma^2(1+\sqrt{5}r+\frac{5}{3}r^2)exp[-\sqrt{5}r]$
# 
# where $r$ is the Euclidean distance and $\sigma^2$ is the variance parameter

# %%
#Fit models using different kernels
model_constant = run_gpflow_scipy(x_train, y_train, gpflow.kernels.Constant())
model_linear = run_gpflow_scipy(x_train, y_train, gpflow.kernels.Linear())
model_RBF = run_gpflow_scipy(x_train, y_train, gpflow.kernels.RBF())
model_cosine = run_gpflow_scipy(x_train, y_train, gpflow.kernels.Cosine())
model_M12 = run_gpflow_scipy(x_train, y_train, gpflow.kernels.Matern12())
model_M32 = run_gpflow_scipy(x_train, y_train, gpflow.kernels.Matern32())
model_M52 = run_gpflow_scipy(x_train, y_train, gpflow.kernels.Matern52())

# %% [markdown]
# #### 10.1.4.2.1 Constant

# %%
model = {'Constant':model_constant}

# %% [markdown]
# Train

# %%
plot_models(model, x_train, y_train,0,2)

# %% [markdown]
# Test

# %%
plot_models(model, x_test, y_test,0,2)

# %% [markdown]
# #### 10.1.4.2.2 Linear

# %%
model = {'Linear':model_linear}

# %% [markdown]
# Train

# %%
plot_models(model, x_train, y_train,0,2)

# %% [markdown]
# Test

# %%
plot_models(model, x_test, y_test,0,2)

# %% [markdown]
# #### 10.1.4.2.3 RBF

# %%
model = {'RBF':model_RBF}

# %% [markdown]
# Train

# %%
plot_models(model, x_train, y_train,0,2)

# %% [markdown]
# Test

# %%
plot_models(model, x_test, y_test,0,2)

# %% [markdown]
# #### 10.1.4.2.4 Cosine

# %%
model = {'Cosine':model_cosine}

# %% [markdown]
# Train

# %%
plot_models(model, x_train, y_train,0,2)

# %% [markdown]
# Test

# %%
plot_models(model, x_test, y_test,0,2)

# %% [markdown]
# #### 10.1.4.2.5 Matern

# %%
model = {'Matern12':model_M12,'Matern32':model_M32,'Matern52':model_M52}

# %% [markdown]
# Train

# %%
plot_models(model, x_train, y_train,0,2)

# %% [markdown]
# Test

# %%
plot_models(model, x_test, y_test,0,2)

# %% [markdown]
# #### 10.1.4.2.6 Analyze Model Predictions

# %% [markdown]
# All Kernels

# %%
model = {'Constant':model_constant,'Linear':model_linear,'RBF':model_RBF,'Cosine':model_cosine,'Matern12':model_M12,'Matern32':model_M32,'Matern52':model_M52}

# %%
plot_function(model,train,test)

# %% [markdown]
# RBF and Matern Kernels

# %%
model = {'RBF':model_RBF,'Matern12':model_M12,'Matern32':model_M32,'Matern52':model_M52}

# %%
plot_function(model,train,test)

# %% [markdown]
# RBF, Matern32, and Matern52 Kernels
# 
# Note: All three of these kernels give very small variances.

# %%
model = {'RBF':model_RBF,'Matern32':model_M32,'Matern52':model_M52}

# %%
plot_function(model,train,test)

# %% [markdown]
# ### 10.1.4.3 What happens if we use different train/test fractions?
# 
# The example above used an 80/20 train/test split. How much data do we need to train with using the RBF and Matern kernels?

# %% [markdown]
# #### 10.1.4.3.1 50/50 Split

# %%
x_train, y_train, x_test, y_test = shuffle_and_split(data, n_params, fraction_train=0.5)
# Fit model
model_RBF = run_gpflow_scipy(x_train, y_train, gpflow.kernels.RBF(lengthscales=np.ones(n_params)))
model_M12 = run_gpflow_scipy(x_train, y_train, gpflow.kernels.Matern12())
model_M32 = run_gpflow_scipy(x_train, y_train, gpflow.kernels.Matern32())
model_M52 = run_gpflow_scipy(x_train, y_train, gpflow.kernels.Matern52())

# %%
model = {'RBF':model_RBF,'Matern12':model_M12,'Matern32':model_M32,'Matern52':model_M52}

y_train.shape = (12,1)
y_test.shape = (13,1)
train = np.concatenate((x_train,y_train),axis=1)
test = np.concatenate((x_test,y_test),axis=1)
 
plot_function(model,train,test)

# %% [markdown]
# #### 10.1.4.3.2 25/75 Split

# %%
x_train, y_train, x_test, y_test = shuffle_and_split(data, n_params, fraction_train=0.25)
# Fit model
model_RBF = run_gpflow_scipy(x_train, y_train, gpflow.kernels.RBF(lengthscales=np.ones(n_params)))
model_M12 = run_gpflow_scipy(x_train, y_train, gpflow.kernels.Matern12())
model_M32 = run_gpflow_scipy(x_train, y_train, gpflow.kernels.Matern32())
model_M52 = run_gpflow_scipy(x_train, y_train, gpflow.kernels.Matern52())

# %%
model = {'RBF':model_RBF,'Matern12':model_M12,'Matern32':model_M32,'Matern52':model_M52}

y_train.shape = (6,1)
y_test.shape = (19,1)
train = np.concatenate((x_train,y_train),axis=1)
test = np.concatenate((x_test,y_test),axis=1)
 
plot_function(model,train,test)

# %% [markdown]
# #### 10.1.4.3.3 10/90 Split

# %%
x_train, y_train, x_test, y_test = shuffle_and_split(data, n_params, fraction_train=0.1)
# Fit model
model_RBF = run_gpflow_scipy(x_train, y_train, gpflow.kernels.RBF(lengthscales=np.ones(n_params)))
model_M12 = run_gpflow_scipy(x_train, y_train, gpflow.kernels.Matern12())
model_M32 = run_gpflow_scipy(x_train, y_train, gpflow.kernels.Matern32())
model_M52 = run_gpflow_scipy(x_train, y_train, gpflow.kernels.Matern52())

# %%
model = {'RBF':model_RBF,'Matern12':model_M12,'Matern32':model_M32,'Matern52':model_M52}

y_train.shape = (2,1)
y_test.shape = (23,1)
train = np.concatenate((x_train,y_train),axis=1)
test = np.concatenate((x_test,y_test),axis=1)
 
plot_function(model,train,test)

# %% [markdown]
# ## 10.1.5 Apply GPflow code to Breakout Time Dataset

# %% [markdown]
# **Objective**: Use GPflow to predict breakout time given five parameters (thickness, laser energy, Be gamma, wall opacity, and flux limiter)
# 
# Kernels: RBF and Matern

# %% [markdown]
# ### 10.1.5.1 Load in Dataset as a dataframe

# %%
csv_path = '/home/zpqls/workspace/CRASHBreakout.csv'

# %%
df = pd.read_csv(csv_path)
df = df.drop(['cmeasure.1','measure.2','measure.3'],axis=1)
df.columns = ["thickness",
                "laser_energy",
                "Be_gamma",
                "wall_opacity",
                "flux_limiter",
                "breakout_time"]

# %%
pd.options.display.max_rows=104
df

# %% [markdown]
# ### 10.1.5.2 Split into training and test sets and fit GP model without normalizing
# 
# I was curious how this would look

# %%
n_params=5
x_train, y_train, x_test, y_test = shuffle_and_split(df, 5, fraction_train=0.8)

# %%
# Fit model
model_RBF = run_gpflow_scipy(x_train, y_train, gpflow.kernels.RBF(lengthscales=np.ones(n_params)))
model_M12 = run_gpflow_scipy(x_train, y_train, gpflow.kernels.Matern12(lengthscales=np.ones(n_params)))
model_M32 = run_gpflow_scipy(x_train, y_train, gpflow.kernels.Matern32(lengthscales=np.ones(n_params)))
model_M52 = run_gpflow_scipy(x_train, y_train, gpflow.kernels.Matern52(lengthscales=np.ones(n_params)))

# %%
model = {'RBF':model_RBF,'Matern12':model_M12,'Matern32':model_M32,'Matern52':model_M52}

# %%
plot_models(model, x_train, y_train,250,550)

# %%
plot_models(model, x_test, y_test,250,550)

# %% [markdown]
# Observation: Without scaling, we can see that none of the GP models give good predictions.

# %% [markdown]
# ### 10.1.5.3 Scaled data

# %%
n_params=5

#Split dataframe into parameters and outputs to do the normalization
param_values = df.values[:, :n_params]
breakout_time_values = df.values[:, n_params]

# %% [markdown]
# Define functions to scale data and apply

# %%
def param_bounds():
    """Return parameter bounds"""

    #units: mm
    bounds_thickness = ( np.asarray(
        [[ 17.5, 22.5 ]]
        
    ))

    #units: kJ/mol
    bounds_laser_energy = (np.asarray(
        [[ 3650. , 4000. ]
        ]
    ))
    
    bounds_Be_gamma = (np.asarray(
        [[ 1.35 , 1.8 ]
        ]
    ))
    
    bounds_wall_opacity = (np.asarray(
        [[ 0.65 , 1.35 ]
        ]
    ))
    
    bounds_flux_limiter = (np.asarray(
        [[ 0.045 , 0.08 ]
        ]
    ))

    bounds = np.vstack((bounds_thickness,bounds_laser_energy,bounds_Be_gamma,bounds_wall_opacity,bounds_flux_limiter))

    return bounds

# %%
def params_real_to_scaled(params, bounds):
    """Convert sample with physical units to values between 0 and 1"""
    return (params - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])

# %%
scaled_param_values = params_real_to_scaled(param_values, param_bounds())

# %%
def breakout_time_bounds():
    """Return the bounds on breakout time in units of ps"""

    bounds = ( np.asarray(
        [ 305.013, 520.389 ],
    ))

    return bounds

# %%
def values_real_to_scaled(values, bounds):
    """Convert breakout time with physical units to value between 0 and 1"""
    return (values - bounds[0]) / (bounds[1] - bounds[0])

# %%
scaled_breakout_time_values = values_real_to_scaled(breakout_time_values, breakout_time_bounds())

# %% [markdown]
# After scaling, combine into a new dataframe.

# %%
scaled_data = np.hstack((scaled_param_values,
                         scaled_breakout_time_values.reshape(-1,1)
                        ))

column_names = ["thickness",
                "laser_energy",
                "Be_gamma",
                "wall_opacity",
                "flux_limiter",
                "breakout_time"]

df_scaled = pd.DataFrame(scaled_data, columns=column_names)

# %%
pd.options.display.max_rows=104
df_scaled

# %% [markdown]
# Split into training and test sets and fit GP model

# %%
n_params=5
x_train, y_train, x_test, y_test = shuffle_and_split(df_scaled, 5, fraction_train=0.8)

# %%
# Fit model
model_RBF = run_gpflow_scipy(x_train, y_train, gpflow.kernels.RBF(lengthscales=np.ones(n_params)))
model_M12 = run_gpflow_scipy(x_train, y_train, gpflow.kernels.Matern12(lengthscales=np.ones(n_params)))
model_M32 = run_gpflow_scipy(x_train, y_train, gpflow.kernels.Matern32(lengthscales=np.ones(n_params)))
model_M52 = run_gpflow_scipy(x_train, y_train, gpflow.kernels.Matern52(lengthscales=np.ones(n_params)))

# %%
model = {'RBF':model_RBF,'Matern12':model_M12,'Matern32':model_M32,'Matern52':model_M52}

# %%
plot_models(model, x_train, y_train,0,1)

# %%
plot_models(model, x_test, y_test,0,1)

# %% [markdown]
# Observation: GP predictions improve when scaling is used. In this case, Matern12 gives the lowest sum of squares error.

# %%


# %% [markdown]
# <!--NAVIGATION-->
# < [10.0 Gaussian Process Emulators and Surrogate Models](https://ndcbe.github.io/cbe67701-uncertainty-quantification/10.00-Gaussian-Process-Emulators-and-Surrogate-Models.html) | [Contents](toc.html) | [10.2 A simple example of Bayesian quadrature](https://ndcbe.github.io/cbe67701-uncertainty-quantification/10.02-Bayesian-quadrature.html)<p><a href="https://colab.research.google.com/github/ndcbe/cbe67701-uncertainty-quantification/blob/master/docs/10.01-Contributed-Example.ipynb"> <img align="left" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" title="Open in Google Colaboratory"></a><p><a href="https://ndcbe.github.io/cbe67701-uncertainty-quantification/10.01-Contributed-Example.ipynb"> <img align="left" src="https://img.shields.io/badge/Github-Download-blue.svg" alt="Download" title="Download Notebook"></a>


