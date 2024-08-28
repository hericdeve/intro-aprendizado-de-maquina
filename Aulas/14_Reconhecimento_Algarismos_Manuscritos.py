#=======================================================================
# EXPERIMENTO 11.1 
#
#   RECONHECIMENTO DE DIGITOS MANUSCRITOS UTILIZANDO
#     - CLASSIFICADORES BAYESIANOS
#     - REGRESSÃO LOGÍSTICA
#     - CALSSIFICADOR KNN
#=======================================================================

#-----------------------------------------------------------------------
# IMPORTAÇÃO DE BIBLIOTECAS DO SCIKIT-LEARN
#-----------------------------------------------------------------------

import pandas as pd
from matplotlib import pyplot as plt

from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

#-----------------------------------------------------------------------
# IMPORTAR O CONJUNTO DE DADOS EM UM DATAFRAME PANDAS 
#-----------------------------------------------------------------------

dados = pd.read_excel("Digits.xlsx")
dados = dados.iloc[:,1:]

#-----------------------------------------------------------------------
# OBTER A MATRIZ X (ATRIBUTOS) E O VETOR Y (ALVO) 
#-----------------------------------------------------------------------

x = dados.iloc[:,:-1].to_numpy()
y = dados.iloc[:,-1].to_numpy()

#-----------------------------------------------------------------------
# VISUALIZAR ALGUNS DÍGITOS 
#-----------------------------------------------------------------------

for i in range(10):
    
    plt.figure(figsize=(40,240))
    imagem = plt.subplot(1,10,i+1)
    imagem.set_title("y = %d" % y[i])
    imagem.imshow(
        x[i,:].reshape(8,8),
        interpolation='nearest',
        cmap='binary',
        vmin=0, vmax=16
        )

plt.show()
    
#-----------------------------------------------------------------------
# EMBARALHAR AS AMOSTRAS PARA EVITAR EVENTUAL VIÉS DE SELEÇÃO 
#-----------------------------------------------------------------------

dados = dados.sample(frac=1)

x = dados.iloc[:,:-1].to_numpy()
y = dados.iloc[:,-1].to_numpy()

#-----------------------------------------------------------------------
# CLASSIFICADOR BAYESIANO COM DISTRIBUIÇÃO MULTINOMIAL
#-----------------------------------------------------------------------

print ( " ")
print ( " CLASSIFICADOR BAYESIANO COM DISTRIBUIÇÃO MULTINOMIAL ")
print ( " ")

classificador = MultinomialNB(alpha=1.0)
y_pred = cross_val_predict(classificador,x,y,cv=5)

print ( "Acurácia = %6.4f" % accuracy_score(y,y_pred) )

#-----------------------------------------------------------------------
# CLASSIFICADOR BAYESIANO COM DISTRIBUIÇÃO GAUSSIANA
#-----------------------------------------------------------------------

print ( " ")
print ( " CLASSIFICADOR BAYESIANO COM DISTRIBUIÇÃO GAUSSIANA ")
print ( " ")

classificador = GaussianNB()
y_pred = cross_val_predict(classificador,x,y,cv=5)

print ( "Acurácia = %6.4f" % accuracy_score(y,y_pred) )

#-----------------------------------------------------------------------
# REGRESSÃO LOGÍSTICA
#-----------------------------------------------------------------------

print ( " ")
print ( " REGRESSÃO LOGÍSTICA ")
print ( " ")

for k in range(-4,3):
    c = pow(10,k)
    classificador = LogisticRegression(penalty='l2',C=c,max_iter=10000)
    y_pred = cross_val_predict(classificador,x,y,cv=5,n_jobs=5)

    print ( "C = %14.6f  --->  Acurácia = %6.4f" % (c,accuracy_score(y,y_pred) ) )

#-----------------------------------------------------------------------
# CLASSIFICADOR KNN
#-----------------------------------------------------------------------

print ( " ")
print ( " CLASSIFICADOR KNN ")
print ( " ")

for k in range(1,20,2):
    classificador = KNeighborsClassifier(n_neighbors=k)
    y_pred = cross_val_predict(classificador,x,y,cv=5,n_jobs=5)

    print ( "k = %2d  --->  Acurácia = %6.4f" % (k,accuracy_score(y,y_pred) ) )


classificador = KNeighborsClassifier(n_neighbors=1)
y_pred = cross_val_predict(classificador,x,y,cv=5,n_jobs=5)

# print ( " ")
# print ( "Acurácia = %6.4f" % accuracy_score(y,y_pred))

# print ( " ")
# print ( "Matriz de Confusão:")
# print ( " ")

# print ( confusion_matrix(y,y_pred) )

# plt.figure(figsize=(4,4))
# imagem = plt.subplot(1,1,1)
# imagem.set_title("y = %d" % y[24])
# imagem.imshow(
#     x[i,:].reshape(8,8),
#     interpolation='nearest',
#     cmap='binary',
#     vmin=0, vmax=16
#     )
# plt.show()

    