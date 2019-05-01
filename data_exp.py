import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


gauss_data = pd.read_csv('twoGaussians.csv', header=None)

# Get data in a format that fits sklearn
gauss_data[11] = pd.Categorical(gauss_data[11])
gauss_data[11] = gauss_data[11].cat.codes
X_raw = gauss_data.values

y = X_raw[:, -1]
X = X_raw[:, :-1]

#Data Explore

m_1 = np.mean(X[:5000, :])
m_2 = np.mean(X[5000:, :])

s_1 = np.std(X[:5000, :])
s_2 = np.std(X[5000:, :])

#covar_1 = np.cov(X[:5000, :])
#covar_2 = np.cov(X[5000:, :])

print("****\t Mean  - Class One {} Class Two {} \t****\n".format(m_1, m_2))
print("****\t STD   - Class One {} Class Two {} \t****\n".format(s_1, s_2))
