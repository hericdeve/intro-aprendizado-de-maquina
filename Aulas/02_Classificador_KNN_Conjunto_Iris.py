#===============================================================================
# EXPERIMENTO 02 - CLASSIFICADOR KNN PARA O CONJUNTO IRIS
#===============================================================================

#-------------------------------------------------------------------------------
# Importar bibliotecas
#-------------------------------------------------------------------------------

import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

#-------------------------------------------------------------------------------
# Ler o arquivo CSV com os dados do conjunto IRIS
#-------------------------------------------------------------------------------

dados = pd.read_csv('Iris_Data.csv')    

#-------------------------------------------------------------------------------
# Embaralhar o conjunto de dados para garantir que a divisão entre os dados de
# treino e os dados de teste esteja isenta de qualquer viés de seleção
#-------------------------------------------------------------------------------

dados_embaralhados = dados.sample(frac=1,random_state=12345)

#-------------------------------------------------------------------------------
# Criar os arrays X e Y para o conjunto de treino e para o conjunto de teste
#-------------------------------------------------------------------------------

# conjunto de treino

x_treino = dados_embaralhados.iloc[:100,:-1].values
y_treino = dados_embaralhados.iloc[:100,-1].values

# conjunto de teste

x_teste = dados_embaralhados.iloc[100:,:-1].values
y_teste = dados_embaralhados.iloc[100:,-1].values

#-------------------------------------------------------------------------------
# Treinar um classificador KNN com o conjunto de treino
#-------------------------------------------------------------------------------

classificador = KNeighborsClassifier(n_neighbors=10)

classificador = classificador.fit(x_treino,y_treino)

#-------------------------------------------------------------------------------
# Obter as respostas do classificador no mesmo conjunto onde foi treinado
#-------------------------------------------------------------------------------

y_resposta_treino = classificador.predict(x_treino)

#-------------------------------------------------------------------------------
# Obter as respostas do classificador no conjunto de teste
#-------------------------------------------------------------------------------

y_resposta_teste = classificador.predict(x_teste)

#-------------------------------------------------------------------------------
# Verificar a acurácia do classificador
#-------------------------------------------------------------------------------

print ("\nDESEMPENHO DENTRO DA AMOSTRA DE TREINO\n")

total   = len(y_treino)
acertos = sum(y_resposta_treino==y_treino)
erros   = sum(y_resposta_treino!=y_treino)

print ("Total de amostras: " , total)
print ("Respostas corretas:" , acertos)
print ("Respostas erradas: " , erros)

acuracia = acertos / total

print ("Acurácia = %.1f %%" % (100*acuracia))

print ("\nDESEMPENHO FORA DA AMOSTRA DE TREINO\n")

total   = len(y_teste)
acertos = sum(y_resposta_teste==y_teste)
erros   = sum(y_resposta_teste!=y_teste)

print ("Total de amostras: " , total)
print ("Respostas corretas:" , acertos)
print ("Respostas erradas: " , erros)

acuracia = acertos / total

print ("Acurácia = %.1f %%" % (100*acuracia))

#-------------------------------------------------------------------------------
# Verificar a variação da acurácia com o número de vizinhos
#-------------------------------------------------------------------------------

print ( "\n  K TREINO  TESTE")
print ( " -- ------ ------")

for k in range(1,15):

    classificador = KNeighborsClassifier(n_neighbors=k)
    classificador = classificador.fit(x_treino,y_treino)

    y_resposta_treino = classificador.predict(x_treino)
    y_resposta_teste  = classificador.predict(x_teste)
    
    acuracia_treino = sum(y_resposta_treino==y_treino)/len(y_treino)
    acuracia_teste  = sum(y_resposta_teste ==y_teste) /len(y_teste)
    
    print(
        "%3d"%k,
        "%6.1f" % (100*acuracia_treino),
        "%6.1f" % (100*acuracia_teste)
        )
    
    


















