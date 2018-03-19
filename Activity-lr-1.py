
# coding: utf-8

# 
# # Linear Regression Example
# 
# This example uses the truck sales dataset to illustrate ordinary least-squares (OLS), or linear regression. The
# plot shows the line that linear regression learns, which best minimizes the residual sum of squares
# between the observed responses in the dataset, and the responses predicted by
# the linear approximation. We also compute the residual sum of squares and the variance score for the model.
# 
# 

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

# Get data
df = pd.read_csv(
    filepath_or_buffer='data/trucks.csv', 
    header=None)

data = df.iloc[:,:].values

print(data[:,0])
X = data[:,0].reshape(-1, 1)
Y = data[:,1].reshape(-1, 1)

# Train the model using the training sets
regr = linear_model.LinearRegression()
regr.fit(X, Y)
slope = regr.coef_[0][0]
intercept = regr.intercept_

print("y = %f + %fx" %(intercept, slope))
print("Mean squared error: %f"
      % np.mean((regr.predict(X) - Y) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %f' % regr.score(X, Y))

# Plot outputs
plt.scatter(X, Y,  color='black')
plt.plot(X, regr.predict(X), color='red',
          linewidth=1)
plt.show()


# In the cell below, we load a subset of the Iris dataset from UCI, specifically all the samples for the "Iris Setosa" flower. The function `model` finds the OLS model for a pair of features in the data and computes the residual sum of squares and the variance score for that model. The parameters `v1` and `v2` are the names of the X and Y variables.

# In[2]:


import numpy as np
import pandas as pd
from sklearn import linear_model

df = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', 
    header=None)

data = df.iloc[:,:].values
data = data[data[:,4] == "Iris-setosa"][:,:4]


def model(X, Y, v1="A", v2="B"):
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    slope = regr.coef_[0][0]
    intercept = regr.intercept_[0]

    print("%s = %f + %fx%s" %(v2, intercept, slope, v1))
    sse = np.sum((regr.predict(X) - Y) ** 2)
    print("Sum of squared errors: %f" % sse)
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %f' % regr.score(X, Y))
    return slope, intercept, sse, v1, v2


# Exercise
# ======
# 
# The samples have 4 features. For each combination of features (each pair or features), consider one of the variables as predictor and the other as response and use the function `model` to find the OLS model that best fits the data. Report the model with the smallest SSE score.

# In[20]:


least = 1
slope_val = 0
intercept_val = 0
for i in range(0, 3):
    for j in range(i+1, 4):
        x = model(data[:,i], data[:,j])
        if(least > x[2]):
            least = x[2]
            slope_val = x[0]
            intercept_val = x[1] 


print("\nModel with minimum SSE has slope as " + str(slope_val) + ", intercept as " + str(intercept_val) + " and SSE as " + str(least))

