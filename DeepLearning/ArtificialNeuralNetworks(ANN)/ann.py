# Artificial Neural Network

# Installing Keras
# Enter the following command in a terminal (or anaconda prompt for Windows users): conda install -c conda-forge keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
#As tres primeiras colunas do dataset nao sao consideradas como variaveis idependentes
X = dataset.iloc[:, 3:13].values
#Queremos prever a partir das informacoes se um cliente tende a sair ou nao do banco
y = dataset.iloc[:, 13].values
# Encoding categorical data
print('------------------------------------------------------------------------')
print(X[0:10,:])
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
#Passando para numeros a variavel idependente de paises
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
#Passando para numeros a coluna de genero - nao precisaremos transformar em dummy variables pois ja estara em formato binario
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
print('------------------------------------------------------------------------')
print(X[0:10,:])
onehotencoder = OneHotEncoder(categorical_features = [1])
#passando para dummy variables a variavel independente da geografia
X = onehotencoder.fit_transform(X).toarray()
#X = X.astype('int')
print('------------------------------------------------------------------------')
print(X[0:10,:])
X = X[:, 1:]
print('------------------------------------------------------------------------')
print(X[0:10,:])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
#modulo responsavel por inicializar a rede
from keras.layers import Dense
#modulo responsavel por gerar as camadas da rede

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
#Utilizaremos a rectifier como funcao de ativacao da camada invisivel que por experimentacao foi identificada como melhor
#e como queremos uma probabilidade do cliente deixar ou nao o banco usaremos um sigmoid function na camada de saida 
#metodo add permite adicionar diferentes camadas
#Dense ira gerar a camada - no caso ja estamos gerando a primeira camada invisivel - possui como parametros quantas saidas possui essa camada - por experimentacao
#e definido utilizando o metodo de K-folder cross validation. Quando nao temos muita certeza tomamos por experimentos realizados a media das entradas/saidas no caso sao 11 entradas + 1 saida = 12/2 = 6 outputs 
#segundo parametro consiste em difnir como sera inicializado os pesos usando uma distribuicao uniforme para inicializa-lo e proximo de 0 (numeros pequenos)
#O tipo de funcao de ativacao
#E o numero de entradas
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
#Nao precisa espificar mais as entradas pois ja identifica automaticamente
#Essa camada nao faz tanta diferença assim, foi mais para tentar amplificar os resultados da anterior e saber como aplicar mais camadas invisiveis
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
#Camada de saida so possuira uma saida que e categorica tendo somente duas categorias, se fossem 3 categorias ou mais ai o numero de saidas e o numero de categorias
#funcao de ativacao ira mudar, sendo a sigmoid que retorna probabilidade
#set tivesse mais de 2 categorias a saida final teria que usar funcao de ativacao soft max function
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
#primeiro parametro e funcao responsavel por calcular os pesos sendo uma boa a funcao adam
#segunda e a funcao utilizada para minimizar o custo, nao sera a do somatorio da diferencia ao quadrado dos residuos mas sim a logarithmic loss que a mais utilizada na gradiente descendente estoistática
#Para variaveis dependetes binarias de duas categorias e binary_crossentropy e se for mais de 2 fica so categorical_crossentropy
#O ultimo e um parametro de validacao, usaremos o metodo de accuracy/precisao para avaliar nosso modelo, isso que faz com que calcule a precisao do modelo e printando no terminal e foi usado os polchetes porque podemos colocar uma lista de metodos se quisermos
#accuracy = numero de previsoes corretas/ numero total de previsoes
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
#Batch/lote dos dados que irao sendo alimentado no algoritmo de gradiente descendente estotástico e o segundo parametro seria o numero de repeticoes do processo chamado de epocas
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)#se y_pred for maior que 0.5/50% retorna verdadeiro - 1

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('Accuracy = ',((cm[0,0]+cm[1,1])/2000)*100,"%")