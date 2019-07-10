import numpy as np
#Biblioteca numerica do python - manipulacoes numericas
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
#Biblioteca utilizada para plotar graficos
import pandas as pd
#Biblioteca para importar e gerenciar dataset
 
#Função da pandas para ler tabela csv
dataset = pd.read_csv('Data.csv')
print("Dataset = \n",dataset)

 
#iloc permite acessar uma coluna somente dando seu indice
#vamos pegar todas as linhas e todas as colunas exceto a ultima
#values significa que estaremos pegando todos os valores
#X sera uma matriz que ira guardar todos os valores de variaveis idependentes
X = dataset.iloc[:, :-1].values
#agora pegaremos as variaveis dependentes
y = dataset.iloc[:,3].values

print("\nSeparando em variaveis idependentes e dependentes:\n")
print("X = \n",X)
print("y = \n",y)

#Lidando com os dados faltantes
#Substituir o dado faltantes pela média da coluna dele

from sklearn.preprocessing import Imputer
#Biblioteca sklearn contém metodos de classes, importaremos
#o modulo de preprocessamento, sendo especificamente a classe Imputer
#Imputer utilaza o método mean(media) para substituir os dados do tipo NaN, substituindo
#pela media da coluna do dado( axis = 0 ) se quiser da linha axis = 1
#Criamos um objeto dele entao com tais atributos
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis=0)
#Para guardar a operacao que ele vai fazer
imputer = imputer.fit(X[:,1:3])
#Para substitui na matrix temos que fazer
X[:, 1:3] = imputer.transform(X[:,1:3])
print("\nSubstituindo dados NaN:\n")
print("X = \n",X)

from sklearn.preprocessing import LabelEncoder
#Outra classe que importaremos para começar a trabalhar com objetos label (rotulados)
#Criando o objeto do tipo LabelEncoder
labelencoder_X = LabelEncoder()
#Vamos rotular os paises, passarao a ser numeros quando aplicar o objeto
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
print("\nRotulando variaveis idependentes:\n")
print("X = \n",X)
#Agora que passamos nossos paises para numeros, temos um problema que é
#que a minha ML pode achar que um pais pode ter um peso maior que o outro
#logo devemos arrumar isso, fazendo o que chama de Dummy Enconding
#que é dividir os dados em tres novas colunas, uma para cada pais em
#que somente contem 1 ou 0, se pertence ou nao aquele pais
from sklearn.preprocessing import OneHotEncoder
#Criamos o objeto da classe e ja definimos que ele vai usar como parametro a coluna 0
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
print("\nPassando para dummy variables:\n")
#print("X = \n",X)
#Passando valores para inteiro para fica melhor de ver
X_int = X.astype('int')
print("X = \n",X_int)

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
print("\nRotulando variaveis dependentes:\n")
print("y = \n",y)

#Biblioteca que utilizaremos para separar o dataset e training set e test_set
from sklearn.model_selection import train_test_split 
#test_size o quanto queremos do dataset que seja testset e o random state para pegar de forma variada
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0   )

print("\nSeparando conjuto de treino e teste:\n")
print("X_train = \n",X_train)
print("X_test = \n",X_test)
print("y_train = \n",y_train)
print("y_test = \n",y_test)

#classe utilizada para feature scaling
from sklearn.preprocessing import StandardScaler
#Variavel para escalar X, criamos seu objeto (por padrao usa a normalizacao para escalonar)
sc_X  = StandardScaler()
#vamos aplicar o objeto agora
#escalando o traing set - sempre usar fit_transform para training set
X_train = sc_X.fit_transform(X_train)
#nao precisa usar o fit para o testing set
X_test = sc_X.transform(X_test)
#Temos que escalar as variaveis rotuladas (paises) ? (ja parecem estar escaladas)
#Depende do contexto, nesse caso nao vai fazer diferencia
#Nesse caso nao precisamos escalonar para as variaveis dependentes, so se
#fosse muito grande e com uma escala de variacao tambem muito grande
print("\nPadronizando escala de variaveis:\n")
print("x_train = \n",X_train)
print("x_test = \n",X_test)


