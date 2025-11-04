#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-11-04T07:43:41.471Z
"""

# # 📖 INSTRUCTION
# 1. [Linear Regression](#1)
# 2. [Multiple Linear Regression](#2)
# 3. [Polynomial Lineer Regression](#3)
# 4. [Decision Tree Regression](#4)
# 5. [Random Forest Regression](#5)
# 6. [Evaluation Regression Models](#6)


# # 📊 Data Exploration
# 
# ## Datasets Used:
# 1. **Linear Regression** - Experience vs Salary
# 2. **Multiple Linear Regression** - Experience, Age vs Salary  
# 3. **Polynomial Regression** - Car Price vs Max Speed
# 4. **Decision Tree Regression** - Tribune Level vs Price
# 5. **Random Forest Regression** - Tribune Level vs Price
# 
# ## What is Regression?
# Regression is a statistical method for modeling relationships between dependent and independent variables to predict continuous values.


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# data visualization
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "notebook"
import seaborn as sns
# machine learning
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# <a id = "1"></a>
# ## Linear Regression


# Import data
linear_regression = pd.read_csv("/kaggle/input/linear-regression-dataset/linear_regression_dataset.csv", sep = ";")

linear_regression.head()

linear_regression.info()

# Plot data
plt.figure(figsize=(10, 6))
plt.scatter(linear_regression.deneyim, linear_regression.maas , s=75)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.title("line fit or linear regression")

# <p>
# y = maas <br>
# x = deneyim <br>
# <br>    
# y = b0 + b1*x <br>
# b0 = constant(bias) <br>
# b1 = coeff , b1 = y/x<br>    
# </p>


# **MSE (Mean Squared Error):**
# 
# $$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$
# 
# Where:
# - $y_i$ = Actual value
# - $\hat{y}_i$ = Predicted value
# - $n$ = Number of samples


# Linear regression model
linear_reg = LinearRegression()

x = linear_regression.deneyim.values.reshape(-1,1)
y = linear_regression.maas.values.reshape(-1,1)

linear_reg.fit(x,y)

# Plot data with fit line

y_head = linear_reg.predict(x)

plt.scatter(x,y)
plt.plot(x,y_head,color = "red")

# Find b0
b0 = linear_reg.predict([[0]]) # manuel
print("b0: ",b0)

b0 = linear_reg.intercept_ # library
print("b0: ",b0) # intercept

# Find b1
b1 = linear_reg.coef_
print("b1: ",b1) # slope

# <p>
# y = b0 + b1*x <br>
# maas = b0 + b1*deneyim <br>
# maas = 1663 + 1138*deneyim<br>     
# </p>


# Average salary for employees with 11 years of experience

maas_yeni = 1663 + 1138*11
print("11 yıl deneyimli ortalama eleman maaşı: ",maas_yeni) # manuel

print("11 yıl deneyimli ortalama eleman maaşı: ",linear_reg.predict([[11]])) # library

# <span style="color:red; font-weight:bold;">There is no need to find b0 or b1 for predict</span>
# 


# <a id = "2"></a>
# ## Multiple Linear Regression


# Import data
multiple_linear_regression = pd.read_csv("/kaggle/input/multiple-linear-regression-dataset/multiple_linear_regression_dataset.csv", sep = ";")

multiple_linear_regression.head()

multiple_linear_regression.info()

# Plot data
plt.figure(figsize=(10, 6))
plt.scatter(multiple_linear_regression['deneyim'], 
            multiple_linear_regression['maas'], 
            c=multiple_linear_regression['yas'], 
            cmap='viridis', 
            s=100, 
            alpha=0.6)
plt.colorbar(label='Yaş')
plt.xlabel('Deneyim (Yıl)', fontsize=12)
plt.ylabel('Maaş', fontsize=12)
plt.title('Deneyim - Maaş İlişkisi (Yaşa Göre Renklendirilmiş)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.show()

# <p>
# simple linear regression = y = b0 + b1*x <br>
# multiple linear regression = y = b0 + b1*x1 + b2*x2 <br>
# <br>    
# maas = dependent variable <br>
# deneyim, yas = independent variable <br>   
# </p>


# Multiple Linear regression model
x = multiple_linear_regression[['deneyim', 'yas']].values
y = multiple_linear_regression.maas.values.reshape(-1,1)

multiple_linear_reg = LinearRegression()
multiple_linear_reg.fit(x,y)

# Find b0
print("b0: ",multiple_linear_reg.intercept_)

# Find b1,b2
print("b1,b2: ",multiple_linear_reg.coef_)

# Predict
multiple_linear_reg.predict(np.array([[10,35],[5,35]])) # Salary of 35-year-olds with 5 and 10 years of experience

# <a id = "3"></a>
# ## Polynomial Lineer Regression


# Import data
polynomial_regression = pd.read_csv("/kaggle/input/polynomial-regression/polynomialregression.csv", sep = ";")

polynomial_regression.head()

polynomial_regression.info()

polynomial_regression.tail()

# Linear regression model
x = polynomial_regression.araba_fiyat.values.reshape(-1,1)
y = polynomial_regression.araba_max_hiz.values.reshape(-1,1)

# Plot data
plt.scatter(x,y)
plt.ylabel("arabanın maksimum hızı")
plt.xlabel("arabanın fiyatı")

# Linear regression model
polynomial_reg = LinearRegression()
polynomial_reg.fit(x,y)

# Predict
y_head = polynomial_reg.predict(x)
print(y_head)

# Plot data with fit line
plt.scatter(x,y)
plt.plot(x,y_head,color="red")
plt.xlabel("arabanın fiyatı")
plt.ylabel("arabanın maksimum hızı")
plt.show()

# Prediction 10.000 value car speed
polynomial_reg.predict([[10000]]) 

# <span style="color:red;">Wrong analysis</span>


# **Polynomial Linear Regression:**
# 
# $$y = b_0 + b_1x + b_2x^2 + b_3x^3 + \ldots + b_nx^n$$
# 
# or in compact form:
# 
# $$y = \sum_{i=0}^{n} b_i x^i$$
# 
# Where:
# - $y$ = Predicted value
# - $b_0$ = Intercept
# - $b_1, b_2, \ldots, b_n$ = Coefficients
# - $x$ = Independent variable
# - $n$ = Degree of polynomial
# 
# ---
# 
# **Examples:**
# 
# **2nd Degree (Quadratic):**
# $$y = b_0 + b_1x + b_2x^2$$
# 
# **3rd Degree (Cubic):**
# $$y = b_0 + b_1x + b_2x^2 + b_3x^3$$


# Polynomial linear regression model
polynomial_regression = PolynomialFeatures(degree = 2)

x_polynomial = polynomial_regression.fit_transform(x)

# Evaluate results
x_polynomial

# Linear regression model
polynomial_regressionn = LinearRegression()

polynomial_regressionn.fit(x_polynomial,y)

# Plot data with fit curve 
y_headd = polynomial_regressionn.predict(x_polynomial)

plt.scatter(x,y)
plt.plot(x,y_headd,color="green",label="Polynomial")
plt.plot(x,y_head,color="red",label="Linear")
plt.legend()
plt.show()

# Higher degree equation fit curve  
polynomial_regressionn = PolynomialFeatures(degree = 4)
x_polynomiall = polynomial_regressionn.fit_transform(x)
polynomial_regressionnn = LinearRegression()
polynomial_regressionnn.fit(x_polynomiall,y)
y_headdd = polynomial_regressionnn.predict(x_polynomiall)

plt.scatter(x,y)
plt.plot(x,y_headd,color="green",label="Polynomial")
plt.plot(x,y_head,color="red",label="Linear")
plt.plot(x,y_headdd,color="blue",label="Higher degree polynomial")
plt.legend()
plt.show()

# <a id = "4"></a>
# ## Decision Tree Regression


# Import data
tree_regression = pd.read_csv("/kaggle/input/decision-tree-regression-dataset/decisiontreeregressiondataset.csv", sep = ";",header=None)

tree_regression.head(10)

tree_regression.info()

# Plot Data
plt.figure(figsize=(10, 6))
plt.scatter(tree_regression[0], tree_regression[1])

# <p>
# <strong>CART:</strong> Classification and Regression Tree<br>
# <strong>Variance:</strong> Measures the spread/disorder in target values (used in regression)<br>
# <strong>Terminal Leaf:</strong> Final node that makes predictions by averaging target values of samples in that node (also called Leaf)<br>
# <strong>Split:</strong> The point where a node divides into child nodes based on a feature threshold<br>
# <strong>Tree Model:</strong> Hierarchical structure that recursively splits data to minimize variance<br>
# <br>
# <strong>Example:</strong> If splitting reduces variance from 50 to 12.5, the Information Gain is 37.5
# </p>
# 
# **Variance (for Regression):**
# $$Variance = \frac{1}{n}\sum_{i=1}^{n}(y_i - \bar{y})^2$$
# 
# **Variance Reduction (Information Gain):**
# $$Reduction = Variance_{parent} - \left[\frac{n_{left}}{n_{total}} \times Variance_{left} + \frac{n_{right}}{n_{total}} \times Variance_{right}\right]$$
# 
# Where:
# - $y_i$ = Each data point
# - $\bar{y}$ = Mean value
# - $n$ = Number of samples
# - $n_{left}$, $n_{right}$ = Number of samples in left and right child nodes


# Decision Tree Regression model
tree_reg= DecisionTreeRegressor()

x = tree_regression.iloc[:,0].values.reshape(-1,1)
y = tree_regression.iloc[:,1].values.reshape(-1,1)

tree_reg.fit(x,y)

# Predict
x_ = np.arange(x.min(),x.max(),0.01).reshape(-1,1)
y_head = tree_reg.predict(x_)

tree_reg.predict([[6]])

# Plot with Predict
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=x.flatten(),
    y=y.flatten(),
    mode='markers',
    name='Actual Data',
    marker=dict(color='orange', size=10)
))

fig.add_trace(go.Scatter(
    x=x_.flatten(),
    y=y_head.flatten(),
    mode='lines',
    name='Prediction',
    line=dict(color='purple', width=3)
))

fig.update_layout(
    title='Decision Tree Regression',
    xaxis_title='Tribun Level',
    yaxis_title='Price',
    template='plotly_white',
    width=900,
    height=600
)

fig.show(config={'displayModeBar': False})

# <a id = "5"></a>
# ## Random Forest Regression


# Import data
random_forest_regression = pd.read_csv("/kaggle/input/random-forest-regression-dataset/randomforestregressiondataset.csv", sep = ";",header=None)

random_forest_regression.head()

# Plot Data
plt.figure(figsize=(10, 6))
plt.scatter(random_forest_regression[0], random_forest_regression[1])

# <p>
# <strong>Random Forest:</strong> Ensemble learning method that combines multiple decision trees<br>
# <strong>Ensemble Learning:</strong> Technique that creates multiple models and combines their predictions for better accuracy<br>
# <strong>Bagging (Bootstrap Aggregating):</strong> Each tree is trained on a random sample (with replacement) from the original data<br>
# <strong>Sub-data:</strong> Random subset of the original dataset used to train each individual tree<br>
# <strong>n Trees:</strong> The forest consists of multiple independent decision trees (tree1, tree2, tree3, ..., tree n)<br>
# <strong>Averaging:</strong> Final prediction is made by averaging the predictions of all trees<br>
# <strong>Result:</strong> More stable and accurate predictions compared to a single decision tree<br>
# <br>
# <strong>Example:</strong> If tree1 predicts 100, tree2 predicts 110, tree3 predicts 105, the final result = (100+110+105)/3 = 105
# </p>
# 
# **Random Forest Prediction:**
# $$\hat{y} = \frac{1}{n}\sum_{i=1}^{n}Tree_i(x)$$
# 
# **Why Random Forest is Better:**
# $$Variance_{single\_tree} > Variance_{random\_forest}$$
# 
# Where:
# - $\hat{y}$ = Final prediction
# - $n$ = Number of trees in the forest
# - $Tree_i(x)$ = Prediction from i-th tree
# - Lower variance = More stable and reliable predictions
# - Each tree sees different random samples, reducing overfitting


# Random Forest Regression model
random_forest_reg= RandomForestRegressor(n_estimators = 100,random_state = 42)

x = random_forest_regression.iloc[:,0].values.reshape(-1,1)
y = random_forest_regression.iloc[:,1].values.ravel()

random_forest_reg.fit(x,y)

# Predict
x_ = np.arange(x.min(),x.max(),0.01).reshape(-1,1)
y_head = random_forest_reg.predict(x_)

print("7.8 seviyesinde fiyatın ne kadar olduğu",random_forest_reg.predict([[7.8]]))

# Plot with Predict
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=x.flatten(),
    y=y.flatten(),
    mode='markers',
    name='Actual Data',
    marker=dict(color='orange', size=10)
))

fig.add_trace(go.Scatter(
    x=x_.flatten(),
    y=y_head.flatten(),
    mode='lines',
    name='Prediction',
    line=dict(color='purple', width=3)
))

fig.update_layout(
    title='Decision Tree Regression',
    xaxis_title='Tribun Level',
    yaxis_title='Price',
    template='plotly_white',
    width=900,
    height=600
)

fig.show(config={'displayModeBar': False})

# <a id = "6"></a>
# ## Evaluation Regression Models


# <p>
# <strong>Residual (Error):</strong> The difference between actual value and predicted value<br>
# <strong>Formula:</strong> residual = y - y_head<br>
# <strong>Square Residual:</strong> Squared error - eliminates negative values and penalizes larger errors more heavily<br>
# <strong>Formula:</strong> square residual = (residual)^2<br>
# <strong>SSR (Sum of Squared Residuals):</strong> Total prediction error of the model - lower is better<br>
# <strong>Formula:</strong> SSR = sum((y - y_head)^2)<br>
# <br>
# <strong>y_avg:</strong> Average of all actual values (used as baseline for comparison)<br>
# <strong>SST (Sum of Squares Total):</strong> Total variance in the data - fixed value for the dataset<br>
# <strong>Formula:</strong> SST = sum((y - y_avg)^2)<br>
# <br>
# <strong>Visual Explanation:</strong> In the graph, blue dots represent actual values, red line shows model predictions, and purple horizontal line indicates the average. Vertical distances represent residuals (errors).<br>
# <br>
# <strong>Example:</strong> If y=18000 and y_head=17500 → residual = 500 → square residual = 250,000
# </p>
# 
# **R² (R-Squared) - Coefficient of Determination:**
# $$R^2 = 1 - \frac{SSR}{SST}$$
# 
# **Interpretation:**
# - $R^2 = 1$ → Perfect prediction (all points on the line, SSR = 0)
# - $R^2 = 0.8$ → Model explains 80% of the variance in the data
# - $R^2 = 0$ → Model is no better than using the mean (SSR = SST)
# - $R^2 < 0$ → Model is worse than using the mean (very poor model!)
# - **The closer R² value is to 1, the better the model performance**
# 
# **Where:**
# - $SSR$ = Model's prediction error (want this LOW)
# - $SST$ = Total variance in data (fixed for dataset)
# - Lower SSR → Higher $R^2$ → Better model
# - $R^2$ shows percentage of variance explained by the model


# Predict with Random Forest Regression
random_forest_reg= RandomForestRegressor(n_estimators = 100,random_state = 42)

x = random_forest_regression.iloc[:,0].values.reshape(-1,1)
y = random_forest_regression.iloc[:,1].values.ravel()

random_forest_reg.fit(x,y)

y_head = random_forest_reg.predict(x)

print(f"R² Score: {r2_score(y,y_head)}")
print(f"Modelin açıkladığı varyans: %{(r2_score(y,y_head)*100)}")

# Predict with Linear Regression
linear_reg = LinearRegression()

x = linear_regression.deneyim.values.reshape(-1,1)
y = linear_regression.maas.values.reshape(-1,1)

linear_reg.fit(x,y)

y_head = linear_reg.predict(x)

print(f"R² Score: {r2_score(y,y_head)}")
print(f"Modelin açıkladığı varyans: %{(r2_score(y,y_head)*100)}")

# # 🎯 Conclusion
# 
# ## Model Comparison
# 
# - **Linear Regression**: Fast, interpretable, best for linear relationships
# - **Multiple Linear Regression**: Handles multiple features effectively
# - **Polynomial Regression**: Captures non-linear patterns, risk of overfitting
# - **Decision Tree**: Flexible, easy to interpret, prone to overfitting
# - **Random Forest**: Most stable, reduces overfitting, generally best performance
# 
# ## Key Insights
# 
# - **R² Score** measures model fit (closer to 1 = better)
# - Tree-based models excel with non-linear data
# - Ensemble methods provide more stable predictions


# # 🔗 References
# 
# ## 📚 My Machine Learning Series
# 
# This notebook is part of a comprehensive Machine Learning series:
# 
# | Notebook | Topics Covered |
# |----------|---------------|
# | 📈 **Regression Models** | Linear, Polynomial, Decision Tree, Random Forest *(Current)* |
# | 🎯 **Classification Models** | [Link](https://www.kaggle.com/code/dandrandandran2093/machine-learning-classifications-models) - Logistic Regression, SVM, KNN, etc. |
# 
# ---
# 
# **Course:** Udemy - MACHINE LEARNING by DATAI TEAM
# 
# **Libraries:** NumPy, Pandas, Matplotlib, Plotly, Scikit-learn