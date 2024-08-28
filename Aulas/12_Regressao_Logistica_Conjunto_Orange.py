#===============================================================================
#
#  EXPERIMENTO 9.1 - CLASSIFICADORES KNN E REGRESSAO LOGISTICA
#                    PARA O CONJUNTO ORANGE
#
#    Vamos construir um modelo preditivo de "churn" para uma empresa
#    de telecomunicações.
#
#===============================================================================

#-------------------------------------------------------------------------------
# Importar bibliotecas
#-------------------------------------------------------------------------------

import pandas as pd

from sklearn.neighbors     import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler

from sklearn.linear_model  import LogisticRegression

from matplotlib import pyplot as plt

#-------------------------------------------------------------------------------
# Ler o arquivo CSV com os dados do conjunto IRIS
#-------------------------------------------------------------------------------

dados = pd.read_csv('Orange_Telecom_Churn_Data.csv')    

#-------------------------------------------------------------------------------
# Explorar os dados
#-------------------------------------------------------------------------------

print ( '\nImprimir o conjunto de dados:\n')

print(dados)

print ( '\nImprimir o conjunto de dados transposto')
print ('para visualizar os nomes de todas as colunas:\n')

print(dados.T)

print ( '\nImprimir os tipos de cada variável:\n')

print(dados.dtypes)

print ( '\nIdentificar as variáveis categóricas:\n')

variaveis_categoricas = [
    x for x in dados.columns if dados[x].dtype == 'object' or x == 'area_code'
    ]

print(variaveis_categoricas)

print ( '\nVerificar a cardinalidade de cada variável categórica:')
print ( 'obs: cardinalidade = qtde de valores distintos que a variável pode assumir\n')

for v in variaveis_categoricas:
    
    print ('\n%15s:'%v , "%4d categorias" % len(dados[v].unique()))
    print (dados[v].unique(),'\n')    

#-------------------------------------------------------------------------------
# Executar preprocessamento dos dados
#-------------------------------------------------------------------------------
  
# state           --> não-ordinal com   51 categorias --> descartar OK
# area_code       --> não-ordinal com    3 categorias --> one-hot encoding OK
# phone_number    --> não-ordinal com 5000 categorias --> descartar OK
# intl_plan       --> binária --> binarizar (mapear para 0/1) OK
# voice_mail_plan --> binária --> binarizar (mapear para 0/1) OK

print ( '\nDescartar as variáveis de cardinalidade muito alta:\n')

print (dados.T)
dados = dados.drop(['state','phone_number'],axis=1)
print (dados.T)

print ( '\nAplicar one-hot encoding nas variáveis que tenham')
print ( '3 ou mais categorias:')

dados = pd.get_dummies(dados,columns=['area_code'])
print (dados.head(5).T)

print ( '\nAplicar binarização simples nas variáveis que tenham')
print ( 'apenas 2 categorias:\n')

binarizador = LabelBinarizer()
for v in ['intl_plan','voice_mail_plan']:
    dados[v] = binarizador.fit_transform(dados[v])
print (dados.head(5).T)

print ( '\nVerificar a quantidade de amostras de cada classe:\n')

print(dados['churned'].value_counts())

print ( '\nVerificar o valor médio de cada atributo em cada classe:')

print(dados.groupby(['churned']).mean().T)

#-------------------------------------------------------------------------------
# Plotar diagrama de dispersão por classe
#-------------------------------------------------------------------------------
  
atributo1 = 'total_day_minutes'
atributo2 = 'total_eve_minutes'

cores = [ 'red' if x else 'blue' for x in dados['churned'] ]

grafico = dados.plot.scatter(
    atributo1,
    atributo2,
    c      = cores,
    s      = 10,
    marker = 'o',
    alpha  = 0.5,
    figsize = (14,14)
    )

plt.show()

#-------------------------------------------------------------------------------
# Selecionar os atributos que serão utilizados pelo classificador
#-------------------------------------------------------------------------------
  
atributos_selecionados = [
    'account_length',
    'intl_plan',
    'voice_mail_plan',
    'number_vmail_messages',
    'total_day_minutes',
    'total_day_calls',
    'total_day_charge',
    'total_eve_minutes',
    'total_eve_calls',
    'total_eve_charge',
    'total_night_minutes',
    'total_night_calls',
    'total_night_charge',
    'total_intl_minutes',
    'total_intl_calls',
    'total_intl_charge',
    'number_customer_service_calls',
    #'area_code_408',
    #'area_code_415',
    #'area_code_510'
    'churned'
    ]

dados = dados[atributos_selecionados]

#-------------------------------------------------------------------------------
# Embaralhar o conjunto de dados para garantir que a divisão entre os dados de
# treino e os dados de teste esteja isenta de qualquer viés de seleção
#-------------------------------------------------------------------------------

dados_embaralhados = dados.sample(frac=1,random_state=12345)

#-------------------------------------------------------------------------------
# Criar os arrays X e Y separando atributos e alvo
#-------------------------------------------------------------------------------

x = dados_embaralhados.loc[:,dados_embaralhados.columns!='churned'].values
y = dados_embaralhados.loc[:,dados_embaralhados.columns=='churned'].values

#-------------------------------------------------------------------------------
# Separar X e Y em conjunto de treino e conjunto de teste
#-------------------------------------------------------------------------------

q = 4000  # qtde de amostras selecionadas para treinamento

# conjunto de treino

x_treino = x[:q,:]
y_treino = y[:q].ravel()

# conjunto de teste

x_teste = x[q:,:]
y_teste = y[q:].ravel()

#-------------------------------------------------------------------------------
# Ajustar a escala dos atributos nos conjuntos de treino e de teste
#-------------------------------------------------------------------------------

ajustador_de_escala = MinMaxScaler()
ajustador_de_escala.fit(x_treino)

x_treino = ajustador_de_escala.transform(x_treino)
x_teste  = ajustador_de_escala.transform(x_teste)

#-------------------------------------------------------------------------------
# Treinar um classificador KNN com o conjunto de treino
#-------------------------------------------------------------------------------

classificador = KNeighborsClassifier(n_neighbors=5)

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

for k in range(1,50,2):

    classificador = KNeighborsClassifier(
        n_neighbors = k,
        weights     = 'uniform',
        p           = 1
        )
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
    
#-------------------------------------------------------------------------------
# Verificar a variação da acurácia para REGRESSAO LOGISTICA
#-------------------------------------------------------------------------------

print ( "\n              C TREINO  TESTE")
print (   " -------------- ------ ------")

for k in range(0,21):
    
    c = pow(10,k/10)

    classificador = LogisticRegression(
        penalty = 'l2',
        C       = c,
        solver  = 'lbfgs'
        )
    classificador = classificador.fit(x_treino,y_treino)

    y_resposta_treino = classificador.predict(x_treino)
    y_resposta_teste  = classificador.predict(x_teste)
    
    acuracia_treino = sum(y_resposta_treino==y_treino)/len(y_treino)
    acuracia_teste  = sum(y_resposta_teste ==y_teste) /len(y_teste)
    
    print(
        "%14.6f"% c,
        "%6.1f" % (100*acuracia_treino),
        "%6.1f" % (100*acuracia_teste)
        )
    












