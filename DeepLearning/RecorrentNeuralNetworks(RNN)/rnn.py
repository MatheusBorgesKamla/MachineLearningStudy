from sklearn.model_selection import train_test_split
from keras.layers import *
from keras.models import Model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras import regularizers
from keras.optimizers import *


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

''' Primeira camada recorrente
Neste exemplo a camada recorrente irá processar todos os tempos de entrada e a saída será o estado h 
do último tempo (batch_size, output_features), gerando uma saída sem a dimensão temporal '''



#O tamanho da entrada X é de 10 váriaveis com 100 tempos
inputs = Input((100, 10))
#Uma camada recorrente com estado h de tamanho 32
rnn = SimpleRNN(32)


out = rnn(inputs)

model = Model(inputs,out)
model.summary()

'''Quando queremos múltiplas saídas que mantenham a dimensão temporal devemos indicar ao Keras essa especifícação. 
Assim temos uma saída (batch_size, timesteps, output_features))'''



inputs = Input((100, 10))
#Return sequences indica para o Keras que a camada deve retornar o estado h em cada tempo
rnn = SimpleRNN(32, return_sequences=True)


out = rnn(inputs)

model = Model(inputs,out)
model.summary()

'''Em vários casos vaos precisar empilhar várias camadas para lidar com problemas complexos. 
Neste caso é necessário definir que cada camada intermediaria deve retornar todos os tempo para a próxima'''

inputs = Input((100, 10))
rnn = SimpleRNN(32, return_sequences=True)
rnn2 = SimpleRNN(32, return_sequences=True)
rnn3 = SimpleRNN(32, return_sequences=True)
rnn4 = SimpleRNN(32)   #A última gera uma saída sem dimensão temporal


x = rnn(inputs)
x = rnn2(x)
x = rnn3(x)
out = rnn4(x)

model = Model(inputs,out)
model.summary()

###################################################

#treinando alguns modelos para fazer a classificação de textos entre positivos e negativos com dataset da IMDB

from keras.datasets import imdb

max_words = 20000 #Iremos tomar aleatoriamente 20000 palavras do dataset do imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words) # já divido entre dataset de treino e de teste

print("X_train length: ", len(x_train))
print("X_test length: ", len(x_test))
#São divididas igualmente, 25000 exemplos 'listas de palavras' entre o treino e o teste, sendo  que o X guardará listas que 
#possuem sequencias de numeros representando palavras e o y guardara seu respectivo sentimento 0 ou 1 
word_to_index = imdb.get_word_index()
#Queremos ver o as palavaras que representa cada némero, sendo assim o imdb.get_word_index() guarda tipo um dicionario
#em que associa uma string/indice a um valor numerico
index_to_word = {v: k for k, v in word_to_index.items()}
#Criamos uma lista que faça o contrario, para cada indice ele guarda a palavra. Ou seja percorro os elementos/numeros do
#word_to_index com a varivel v e difino que na posicao v eu guardarei a key da word_to_index
#Essa forma de percorrer e utilizado para estruturas de dados do tipo dictonary

print(x_train[0])
#inde_to_word[x] o que ira retornar para cada iteracao, x a varivel que guardara os valores percoriddos em x_train[0]
print("\n\n"," ".join([index_to_word[x] for x in x_train[0]]))

print("Min value:", min(y_train), "Max value:", max(y_train))
#0 sentimento negativo e 1 sentimento positivo

import numpy as np

average_length = np.mean([len(x) for x in x_train])
#Cada item da lista pode possuir entre poucas palavras ate muitas palavras, logo calculo uma media
#da quantidade de palavras por item
median_length = sorted([len(x) for x in x_train])[len(x_train) // 2]
#Primeiro ordeno esses comprimentos de palavra de cada item e depois pego o ponto do mediano (// e divisao inteira - divide e pega o piso)

print("Average sequence length: ", average_length)
print("Median sequence length: ", median_length)

max_sequence_length = 180
#Defino um comprimento máximo de palavras para cada sequencia em base na media e mediana

from keras.preprocessing import sequence

#Com o modulo sequence nos preenchemos cada sequencia para possuir ate o numero maximo de palavras
#Preenchendo com 0 sequencias menores que o maximo ou truncando os valores quando maiores que o maximo
#O metodo post ou pre e para quando preencho/trunco no inicio ou no final da sequencia 
x_train = sequence.pad_sequences(x_train, maxlen=max_sequence_length, padding='post', truncating='post')
x_test = sequence.pad_sequences(x_test, maxlen=max_sequence_length, padding='post', truncating='post')

print('X_train shape: ', x_train.shape)

from keras.models import Sequential

from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dense

# Single layer LSTM example

#Dimensao da camada invisivel
hidden_size = 32

#Classe que permite gerar pilhas de camadas linear (uma rede)
sl_model = Sequential()
#o método add adiciona uma nova camada e a classe Embedding é utilizado para criar camadas de entrada em que recebe o numero de entradas e saidas dessa primeira camada
sl_model.add(Embedding(max_words, hidden_size))
#LSTM gera uma camada do tipo LSTM que recebe o numero de saidas, o tipo de funcao de ativacao geralmente a tangente hiperbolica
# o quanto irei desconsiderar dos dados de entrada nessa camada e quanto irei desconsideradar do estado de recorrencia anterior
sl_model.add(LSTM(hidden_size, activation='tanh', dropout=0.2, recurrent_dropout=0.2))
#Adicionando a camanda de saida como uma camada comum densely-connected NN que possuira uma saida e uma funcao de ativacao sigmoid
sl_model.add(Dense(1, activation='sigmoid'))
#.compile ira realizar a configuracao final da rede para que possa sser treinada
sl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

epochs = 3

sl_model.fit(x_train, y_train, epochs=epochs, shuffle=True)
loss, acc = sl_model.evaluate(x_test, y_test)

d_model = Sequential()
d_model.add(Embedding(max_words, hidden_size))
d_model.add(LSTM(hidden_size, activation='tanh', dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
d_model.add(LSTM(hidden_size, activation='tanh', dropout=0.2, recurrent_dropout=0.2))
d_model.add(Dense(1, activation='sigmoid'))
d_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

d_model.fit(x_train, y_train, epochs=epochs, shuffle=True)
d_loss, d_acc = d_model.evaluate(x_test, y_test)

print('Single layer model -- ACC {} -- LOSS {}'.format(acc, loss))
print('Double layer model -- ACC {} -- LOSS {}'.format(d_acc, d_loss))




