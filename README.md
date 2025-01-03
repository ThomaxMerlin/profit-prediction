# Profit Prediction using Linear Regression  
**With Jupyter Notebook**

This Jupyter Notebook demonstrates how to build a linear regression model to predict profit based on features like R&D Spend, Administration, Marketing Spend, and State. The dataset used is `profit.csv`, which contains 50 entries with numerical and categorical features.

---

## **Table of Contents**
1. [Prerequisites](#prerequisites)
2. [Getting Started](#getting-started)
3. [Running the Code](#running-the-code)
4. [Code Explanation](#code-explanation)
5. [Results](#results)
6. [License](#license)

---

## **Prerequisites**
Before running the code, ensure you have the following installed:
- Python 3.x
- Required Python libraries:
  ```bash
  pip install numpy pandas seaborn matplotlib scikit-learn statsmodels jupyter
  ```
- Jupyter Notebook (to run the `.ipynb` file).

---

## **Getting Started**
1. **Clone the Repository**  
   Clone this repository to your local machine:
   ```bash
   git clone https://github.com/your-username/profit-prediction.git
   cd profit-prediction
   ```

2. **Download the Dataset**  
   Ensure the dataset `profit.csv` is in the same directory as the notebook.

3. **Launch Jupyter Notebook**  
   Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
   Open the `.ipynb` file from the Jupyter Notebook interface.

---

## **Running the Code**
1. Open the `.ipynb` file in Jupyter Notebook.
2. Run each cell sequentially to execute the code.

---

## **Code Explanation**
### **1. Import Libraries**
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```
- Libraries used for data manipulation, visualization, and modeling.

### **2. Load and Explore Data**
```python
data = pd.read_csv('profit.csv')
data.head()
data.shape
data.describe()
data.info()
```
- Load the dataset and explore its structure, summary statistics, and data types.

### **3. Data Preprocessing**
```python
np.unique(data['State'])
data = pd.get_dummies(data, drop_first=True)
```
- Convert categorical variables (e.g., `State`) into dummy variables for modeling.

### **4. Correlation Analysis**
```python
data.corr()
sns.pairplot(data, hue="State")
```
- Analyze correlations between features and visualize relationships using pair plots.

### **5. Variance Inflation Factor (VIF)**
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
x = data.drop(columns='Profit')
vif_data = pd.DataFrame()
vif_data["feature"] = x.columns
vif_data["VIF"] = [variance_inflation_factor(x.values, i) for i in range(len(x.columns))]
print(vif_data)
```
- Calculate VIF to check for multicollinearity among features.

### **6. Train-Test Split**
```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
```
- Split the data into training and testing sets.

### **7. Build and Train Model**
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
```
- Train a linear regression model on the training data.

### **8. Evaluate Model**
```python
y_predict = model.predict(x_test)
mean_squared_error(y_test, y_predict)
r2_score(y_test, y_predict)
```
- Evaluate the model using Mean Squared Error (MSE) and R² score.

---

## **Results**
- **Coefficients**: The model coefficients indicate the impact of each feature on profit.
- **Intercept**: The intercept value of the linear regression model.
- **Mean Squared Error (MSE)**: A measure of the model's prediction error.
- **R² Score**: Indicates how well the model explains the variance in the target variable.

---

## **License**
This project is open-source and available under the [MIT License](LICENSE). Feel free to use, modify, and distribute it as needed.

---

## **Support**
If you encounter any issues or have questions, feel free to open an issue in this repository or contact me at [minthukywe2020@gmail.com](mailto:minthukywe2020@gmail.com).

---

Enjoy exploring the profit prediction model in Jupyter Notebook! 🚀
