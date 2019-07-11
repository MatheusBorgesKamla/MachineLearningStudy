# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
#Processo de passar a variável independente categórica de estado para rótulos numéricos
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
#Processo de passar os estados para dummy variables
X = onehotencoder.fit_transform(X).toarray()
#print(X.astype('int'))
# Avoiding the Dummy Variable Trap
#elemino a primeira coluna 1: indica que estou pegando a partir da primeira coluna
#ate o final. Realizo isso para evitar a repetição de dummy variables
X = X[:, 1:]
#print(X.astype('int'))
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#O codigo desenvolvido ate aqui utilizou todas as variaveis independentes
#Porem, podemos aplicar o Bacward Elimination para encontrar um aprendizado mais eficiente
#podendo eliminar certas variaveis independentes que nao influenciam tanto no modelo

import statsmodels.formula.api as sm
#Temos um problema que essa biblioteca nao considera automaticamente as constante b0
#do modelo, logo precisamos adicionar uma coluna de 1 (coeficiente de b0) no inicio de X
X = np.append(np.ones((50,1), int), X, 1 )
#X_opt sera meu X otimizado, irei aplicando o bacward elimination para ir excluindo as variaveis desnecessárias
X_opt = X[:,:]
#Vamos definir o nivel de significancia
sl = 0.05



#Fazendo uma funcao para a backward elimination para isso:

def Backward_Elimination(X, y, sl):
    #Obtendo o numero de variaveis
    num_Var = len(X[0])
    i = 0
    while True:
        #Objeto do modelo de regressao linear da biblioteca stats
        #Ja preenchemos o modelo com as variaveis atuais
        regressor_OLS = sm.OLS(y,X).fit()
        #Comando summary retorna diversos dados estatisticos relacionados as variaveis
        print(regressor_OLS.summary())
        #Pego o maximo valor p dentre as variaveis
        max_p = max(regressor_OLS.pvalues).astype(float)
        #Se for maior que o nivel de significancia eu tenho que exclui-la
        if max_p > sl:
            for j in range(0, num_Var-i):
                if(regressor_OLS.pvalues[j].astype(float) == max_p):
                    #X recebe ele mesmo deletando a coluna j (o 1 indica que e a coluna, para ser a linha seria 0)
                    X = np.delete(X, j, 1)
                    i += 1
        else:
            return X
        
X_opt = Backward_Elimination(X_opt,y,sl)
#Realizando a regressao com o X_opt:
X_train_opt, X_test_opt, y_train_opt, y_test_opt = train_test_split(X_opt, y, test_size = 0.2, random_state = 0)
regressor_opt = LinearRegression()
regressor_opt.fit(X_train_opt, y_train_opt)

# Predicting the Test set results
y_pred_opt = regressor_opt.predict(X_test_opt)

        
    

