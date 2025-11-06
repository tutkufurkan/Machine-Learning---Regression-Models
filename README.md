# Machine Learning Regression Models Tutorial

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Latest-blue.svg)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/sekertutku/Machine-Learning---Regression-Models)

## Overview

This repository provides a comprehensive tutorial on machine learning regression techniques using Python. The project demonstrates 5 essential regression algorithms through practical examples with detailed mathematical explanations, interactive visualizations using Plotly, and model evaluation metrics. Each model is explained with real-world datasets to showcase their unique strengths and use cases.

## 🎮 Interactive Demo

**👉 [Run the Interactive Notebook on Kaggle](https://www.kaggle.com/code/dandrandandran2093/machine-learning-regression-models)**

*For the best experience with interactive Plotly visualizations and pre-configured datasets, use the Kaggle notebook above. All models are ready to run with visual explanations!*

## Table of Contents

- [Introduction](#introduction)
- [Datasets](#datasets)
- [Regression Models](#regression-models)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Key Features](#key-features)
- [Model Performance](#model-performance)
- [Mathematical Foundations](#mathematical-foundations)
- [Contributing](#contributing)
- [References](#references)

## Introduction

Regression analysis is a fundamental statistical method for modeling relationships between dependent and independent variables to predict continuous values. This tutorial covers both classical statistical approaches and modern machine learning techniques, providing a solid foundation for predictive modeling.

## Datasets

The tutorial utilizes 5 different datasets, each designed to demonstrate specific regression concepts:

### 1. Linear Regression Dataset
- **Features**: Experience (deneyim)
- **Target**: Salary (maas)
- **Purpose**: Demonstrates simple linear relationships

### 2. Multiple Linear Regression Dataset
- **Features**: Experience (deneyim), Age (yas)
- **Target**: Salary (maas)
- **Purpose**: Shows multi-variable relationships

### 3. Polynomial Regression Dataset
- **Features**: Car Price (araba_fiyat)
- **Target**: Maximum Speed (araba_max_hiz)
- **Purpose**: Captures non-linear patterns

### 4. Decision Tree Regression Dataset
- **Features**: Tribune Level
- **Target**: Price
- **Purpose**: Hierarchical decision-making patterns

### 5. Random Forest Regression Dataset
- **Features**: Tribune Level
- **Target**: Price
- **Purpose**: Ensemble learning demonstration

## Regression Models

The tutorial covers 5 comprehensive regression techniques:

### 1. Linear Regression
- **Concept**: Fits a straight line through data points
- **Formula**: `y = b₀ + b₁x`
- **Use Case**: Simple relationships between two variables
- **Example**: Predicting salary based on years of experience

**Key Components:**
- **b₀ (Intercept)**: Starting point when x = 0
- **b₁ (Slope)**: Rate of change in y per unit change in x
- **MSE**: Mean Squared Error for error measurement

### 2. Multiple Linear Regression
- **Concept**: Extension to multiple independent variables
- **Formula**: `y = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ`
- **Use Case**: Complex relationships with multiple predictors
- **Example**: Predicting salary based on experience and age

**Key Features:**
- Handles multiple input features simultaneously
- Visualized with 3D scatter plots colored by third variable
- Calculates multiple coefficients for each feature

### 3. Polynomial Regression
- **Concept**: Fits curves to capture non-linear patterns
- **Formula**: `y = b₀ + b₁x + b₂x² + b₃x³ + ... + bₙxⁿ`
- **Use Case**: Non-linear relationships
- **Example**: Predicting car max speed based on price

**Degree Options:**
- **2nd Degree (Quadratic)**: `y = b₀ + b₁x + b₂x²`
- **3rd Degree (Cubic)**: `y = b₀ + b₁x + b₂x² + b₃x³`
- **Higher Degrees**: For more complex curves (risk of overfitting)

**Visualization:** 
- Compares linear vs polynomial fits
- Shows multiple polynomial degrees on same plot

### 4. Decision Tree Regression
- **Concept**: Hierarchical tree structure for predictions
- **Methodology**: CART (Classification and Regression Tree)
- **Use Case**: Complex, non-linear decision boundaries
- **Example**: Tribune level pricing with hierarchical decisions

**Key Concepts:**
- **Variance**: Measures spread in target values
- **Split**: Division point based on feature threshold
- **Terminal Leaf**: Final prediction node
- **Information Gain**: Reduction in variance after split

**Advantages:**
- Easy to interpret and visualize
- Handles non-linear relationships naturally
- No feature scaling required

**Disadvantages:**
- Prone to overfitting
- Unstable to small data changes

### 5. Random Forest Regression
- **Concept**: Ensemble of multiple decision trees
- **Methodology**: Bagging (Bootstrap Aggregating)
- **Use Case**: Robust predictions with reduced overfitting
- **Example**: Enhanced tribune level pricing

**Key Components:**
- **Ensemble Learning**: Combines multiple models
- **Bagging**: Each tree trained on random data sample
- **n_estimators**: Number of trees (100 in this tutorial)
- **Averaging**: Final prediction from averaging all trees

**Advantages:**
- More stable than single decision tree
- Reduces overfitting significantly
- Better generalization
- Lower variance

**Performance:**
- Generally provides best R² scores
- More robust to outliers
- Handles feature interactions automatically

## Requirements

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
plotly>=5.0.0
jupyter>=1.0.0
```

## Installation

### Option 1: Use Kaggle (Recommended) ⭐

The easiest way to explore this tutorial is on Kaggle where everything is pre-configured:

👉 **[Open Interactive Notebook on Kaggle](https://www.kaggle.com/code/dandrandandran2093/machine-learning-regression-models)**

### Option 2: Run Locally

1. Clone the repository:
```bash
git clone https://github.com/sekertutku/Machine-Learning---Regression-Models.git
cd Machine-Learning---Regression-Models
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. **Datasets:**
   
   ✨ All datasets are included in the `input/` directory for reference:
   ```
   input/
   ├── linear_regression_dataset.csv
   ├── multiple_linear_regression_dataset.csv
   ├── polynomialregression.csv
   ├── decisiontreegressiondataset.csv
   └── randomforestregressiondataset.csv
   ```
   
   **Note:** The code uses Kaggle-specific paths (`/kaggle/input/...`). To run locally, update the dataset paths in the code from:
   ```python
   df = pd.read_csv(r"/kaggle/input/dataset-name/file.csv")
   ```
   to:
   ```python
   df = pd.read_csv("input/file.csv")
   ```

## Usage

### On Kaggle (Recommended) ⭐

Simply open the [Kaggle notebook](https://www.kaggle.com/code/dandrandandran2093/machine-learning-regression-models) and run the cells. All dependencies and datasets are pre-configured!

### Locally

#### Running the Complete Tutorial

Execute the main script to run all regression models:

```bash
python machine-learning-regression-models.py
```

#### Running in Jupyter Notebook

For interactive exploration:

```bash
jupyter notebook machine-learning-regression-models.ipynb
```

### Making Predictions

Example usage for each model:

```python
# Linear Regression
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
prediction = linear_reg.predict([[11]])  # Predict for 11 years experience

# Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)

# Decision Tree
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X, y)

# Random Forest
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X, y)
```

## Key Features

### Interactive Visualizations
- **Plotly Integration**: Interactive hover, zoom, and pan capabilities
- **Multiple Plot Types**: Scatter plots, line fits, 3D visualizations
- **Color Mapping**: Visual representation of third variables
- **Comparison Plots**: Side-by-side model comparisons

### Mathematical Explanations
- **LaTeX Formulas**: Clear mathematical notation
- **Step-by-Step Derivations**: Understanding the theory
- **Visual Examples**: Graphs demonstrating concepts
- **Error Metrics**: MSE and R² calculations

### Model Evaluation
- **R² Score (Coefficient of Determination)**:
  - Measures model fit quality
  - Range: -∞ to 1
  - Interpretation: Percentage of variance explained
  - Formula: `R² = 1 - (SSR/SST)`

- **Residual Analysis**:
  - Difference between actual and predicted values
  - Visualized through scatter plots
  - Used for model diagnostics

### Code Quality
- **Clean Structure**: Well-organized and commented code
- **Reusable Functions**: Modular design
- **Error Handling**: Robust implementation
- **Best Practices**: Following scikit-learn conventions

## Model Performance

### Performance Comparison

| Model | Complexity | Speed | Accuracy | Overfitting Risk |
|-------|-----------|-------|----------|-----------------|
| Linear Regression | Low | Very Fast | Good for linear data | Low |
| Multiple Linear Regression | Medium | Fast | Better with multiple features | Low |
| Polynomial Regression | Medium | Fast | Good for curves | Medium-High |
| Decision Tree | High | Fast | High but unstable | High |
| Random Forest | Very High | Moderate | Highest overall | Low |

### When to Use Each Model

**Linear Regression:**
- Clear linear relationship between variables
- Need for interpretability
- Quick baseline model
- Example: Experience vs Salary

**Multiple Linear Regression:**
- Multiple independent variables
- Linear relationships
- Feature importance analysis
- Example: Experience + Age vs Salary

**Polynomial Regression:**
- Non-linear but smooth curves
- Limited complexity (avoid high degrees)
- Example: Price vs Speed with diminishing returns

**Decision Tree:**
- Need for easy interpretation
- Non-linear relationships
- Quick prototyping
- Example: Categorical splits in pricing

**Random Forest:**
- Production-ready predictions
- Need for robustness
- Complex non-linear patterns
- Example: Final pricing model with best accuracy

## Mathematical Foundations

### Linear Regression Formula
```
y = b₀ + b₁x
```
Where:
- y = Predicted value (dependent variable)
- x = Input feature (independent variable)
- b₀ = Intercept (constant/bias)
- b₁ = Slope (coefficient)

### Mean Squared Error (MSE)
```
MSE = (1/n) × Σ(yᵢ - ŷᵢ)²
```
Where:
- n = Number of samples
- yᵢ = Actual value
- ŷᵢ = Predicted value

### R² Score (Coefficient of Determination)
```
R² = 1 - (SSR/SST)
```
Where:
- SSR = Sum of Squared Residuals (model error)
- SST = Total Sum of Squares (total variance)

**Interpretation:**
- R² = 1.0 → Perfect predictions
- R² = 0.8 → Model explains 80% of variance
- R² = 0.0 → Model no better than mean
- R² < 0.0 → Model worse than mean

### Variance Reduction (Decision Trees)
```
Reduction = Variance_parent - [w_left × Variance_left + w_right × Variance_right]
```
Where:
- w_left, w_right = Weight of samples in each branch

### Random Forest Prediction
```
ŷ = (1/n) × Σ Treeᵢ(x)
```
Where:
- n = Number of trees
- Treeᵢ(x) = Prediction from i-th tree

## Key Insights

### Model Selection Strategy
1. **Start Simple**: Begin with Linear Regression as baseline
2. **Add Complexity**: Try Polynomial if non-linear patterns exist
3. **Ensemble Methods**: Use Random Forest for production
4. **Evaluate**: Compare R² scores across models

### Best Practices
- **Feature Scaling**: Normalize features for better performance (except tree-based models)
- **Train-Test Split**: Always validate on unseen data
- **Cross-Validation**: Use k-fold CV for robust evaluation
- **Hyperparameter Tuning**: Optimize polynomial degree, tree depth, n_estimators
- **Avoid Overfitting**: Regularization, pruning, ensemble methods

### Common Pitfalls
- High-degree polynomials causing overfitting
- Single decision trees being unstable
- Not scaling features for linear models
- Ignoring outliers in the data
- Using wrong model for data pattern

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss proposed modifications.

### How to Contribute
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## References

### Course
- **Udemy**: MACHINE LEARNING by DATAI TEAM

### Documentation
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Plotly Python Documentation](https://plotly.com/python/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

### Algorithms & Theory
- [Linear Regression - Scikit-learn](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)
- [Polynomial Features](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
- [Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees)
- [Model Evaluation Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)

### Related Project
- 🎯 **Machine Learning Classification Models** - [[Kaggle]](https://www.kaggle.com/code/dandrandandran2093/machine-learning-classifications-models) [[GitHub]](https://github.com/sekertutku/Machine-Learning---Classifications-Models)

## Acknowledgments

Special thanks to:
- DATAI TEAM for the comprehensive machine learning course
- Scikit-learn developers for excellent ML library
- Plotly team for interactive visualization tools
- The open-source community for making these tools accessible

---

**Note**: This tutorial is intended for educational purposes. The datasets are used for demonstration of regression techniques and may not represent real-world production scenarios.

## 📞 Connect

If you have questions or suggestions:
- Open an issue in this repository
- Connect on [Kaggle](https://www.kaggle.com/dandrandandran2093)
- Visit my website: [tutkufurkan.com](https://www.tutkufurkan.com/)
- Star this repository if you found it helpful!

---

**Happy Learning! 📚✨**

🌐 More projects at [tutkufurkan.com](https://www.tutkufurkan.com/)
