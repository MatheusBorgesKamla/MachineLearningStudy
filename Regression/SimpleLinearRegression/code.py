import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values



#Splitting the dataset into the Training set and Test set
#Nao precisei fazer os passos anteriores de preprocessamento pois
#dados nao necessitam para tratamento

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Nao e necessario o feature scaling pois a biblioteca para regressao linear simples já trata isso automaticamente


#Fitting Simple Linear Regression to the Training Set
#primeiro temos que importar a classe de regressao linear
from sklearn.linear_model import LinearRegression
#inicializando o objeto da classe
regressor = LinearRegression()
#Vamos preencher com os dados de treino
regressor.fit(X_train, y_train)
#A partir daqui ja possuimos nosso modelo, ja foi determinado a relacao entre o X e o y
#Ja foi aplicado o metodo dos minimos quadrados para achar a reta

# Predicting the Test set results
y_pred = regressor.predict(X_test)
#Agora a partir do conjunto de teste X iremos prever o y, utiliza do coef. angular e linear e dos pontos X de teste para achar os y

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
#Estamos passando as coordenadas X e Y dos pontos que queremos plotar e a cor que queremos
#Vermelho sera os pontos reais
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
#Em azul plotaremos a reta prevista para os pontos de treino
plt.title('Salary vs Experience (Training set)')
#Titulo do nosso grafico
plt.xlabel('Years of Experience')
#Titulo do eixo x
plt.ylabel('Salary')
#Titulo do eixo y
plt.show()
#Comando para mostrar na tela

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
