# Polynomial Regression

# Importing the libraries
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
#Ja pegamos a coluna de level, pois ela e a rotulacao dos cargos e usamos 1:2 em vez de so 1 para fazer de X uma matriz
y = dataset.iloc[:, 2].values


# Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Linear Regression to the dataset - utilizando a metodologia de regressao linear no dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
#Aqui escolhemos o grau que queremos utilizar para nosso modelo polinomial
poly_reg = PolynomialFeatures(degree = 4)
#Aqui o que vamos fazer é pegar a variavel idependente em X level e criar novas variaveis idependentes que dependem da primeira de acordo com o grau exponencial
# eu pego meu X q possui somente a variavel a e transformo em 1*b0 + a*b1 + a²*b2 + a³*b3 e assim por diante
X_poly = poly_reg.fit_transform(X)

#poly_reg.fit(X_poly, y)

#Aplicando o modelo de regressao linear para a nova matriz Xpoly com as variaveis idependentes novas formando polinômio
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(X_poly), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
#Essa e uma forma de aumentar o numero de pontos atraves da funcao arange, gerara pontos variando em 0.1 do valor minimo em X ate o maximo
X_grid = np.arange(min(X), max(X), 0.1)
#Passo para matriz novamente, pois a arange retorna um vetor
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
X_poly_new = poly_reg.fit_transform(X_grid)
plt.plot(X_grid, lin_reg_2.predict(X_poly_new), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
print("Para um level 6.5 o metodo de Reg. Linear preve que o salario e ", lin_reg.predict([[6.5]]))

# Predicting a new result with Polynomial Regression
print("Para um level 6.5 o metodo de Reg. Polinomial preve que o salario e ", lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))