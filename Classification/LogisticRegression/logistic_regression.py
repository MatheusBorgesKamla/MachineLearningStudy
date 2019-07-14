# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
#Queremos saber somente se baseando no salário e idade do usuário da rede social se ele compraria ou não um carro de luxo
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,2:4].values
y = dataset.iloc[:,4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
#Matriz que ira conter as previsoes corretas realizadas e tambem as incorretas
#Ler melhor no doc de teoria sobre ela
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix: \n', cm)
#Significa que 65 pessoas que nao comprara foram identificadas corretamente (como pessoas que nao compraram)
#3 pessoas que nao compraram foram identificadas erroneamente como pessosas que compraram
#8 pessoas que compraram foram erroneamente identificadas como que nao compraram
#24 pessoas que nao compraram foram identificadas corretamente como pessoas que nao compraram
#65 + 24 = 89 previsoes corretas e 8 + 3 = 11 incorretas

# Visualising the Training set results
from matplotlib.colors import ListedColormap
#Passando as os dataset de treinos para auxiliares para nao ficar mexendo toda hora
X_set, y_set = X_train, y_train
#np.meshgrid recebe duas matrizes ou vetores e transforma em um plano de coordenadas / grid usando os valores guardado nos vetores passados
#um para cada eixo, fiz esses vetores serem aranges que vão do minimo - 1 até o maximo + 1 (para que nao fique muito em cima os +-1) e variando 
#de 0.01 em 0.01 para que haja bastante pontos
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1 , stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
#A plt.contourf gera o contorno (traço que divide o vermelho do verde) classificando para cada ponto/pixel se ele e vermelho 0 ou verde 1
#np.array gera um array e o reshape remodela a dimensao desse array para do tipo de X1
#o .ravel() retorna os valores de X1 e X2 em 1D dimensao
#alpha seria o tao colorido sera o grid e o cmap as cores que receberao cada ponto do grid fora dos contornos
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.5, cmap = ListedColormap(('red', 'green')))
#plt.xlim(X1.min(), X1.max())
#plt.ylim(X2.min(), X2.max())
#printando os pontos de treino coloridos
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.5, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()