#============================================================================
# EXPERIMENTO 09 - UNDERFITTING E OVERFITTING
#============================================================================

grau_polinomial    =  9    # grau da regressão polinomial

numero_de_amostras = 2000    # número de amostras

#----------------------------------------------------------------------------
# PACOTES E CLASSES IMPORTADAS
#----------------------------------------------------------------------------

import math
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import PolynomialFeatures
from sklearn.linear_model    import LinearRegression
from sklearn.metrics         import mean_squared_error

#----------------------------------------------------------------------------
# DEFINIR UMA FUNÇÃO SENOIDAL BASE PARA GERAR AS AMOSTRAS
#----------------------------------------------------------------------------

x_base = np.linspace(0.00,1.00,num=101).reshape(-1,1)
y_base = np.sin(2*np.pi*x_base)

#----------------------------------------------------------------------------
# GERAR AMOSTRAS ALEATÓRIAS COM DESVIO PADRÃO DE 0.2 EM TORNO DA
# FUNÇÃO SENOIDAL
#----------------------------------------------------------------------------

np.random.seed(0)

x = np.random.rand(numero_de_amostras,1)
y = np.sin(2*np.pi*x) + 0.20*np.random.randn(numero_de_amostras,1)

#----------------------------------------------------------------------------
# DIVIDIR AS AMOSTRAS ENTRE CONJUNTO DE TREINAMENTO E CONJUNTO DE TESTE
#----------------------------------------------------------------------------

x_treino, x_teste, y_treino, y_teste = train_test_split(
    x,
    y,
    test_size = 0.5
    )

#----------------------------------------------------------------------------
# VISUALIZAR AS AMOSTRAS EM UM GRÁFICO DE DISPERSÃO
#----------------------------------------------------------------------------

plt.figure(figsize=(12,12))

plt.title("AMOSTRAS DISPONÍVEIS")

plt.plot(
    x_base,
    y_base,
    color     = 'gray',
    linestyle = 'dotted',
    label     = 'Função-alvo (desconhecida)'
    )

plt.scatter(
    x_treino,
    y_treino,
    color     = 'green',
    marker    = 'o',
    s         = 30,
    alpha     = 0.5,
    label     = 'amostras de treinamento'
    )

plt.scatter(
    x_teste,
    y_teste,
    color     = 'red',
    marker    = 'o',
    s         = 30,
    alpha     = 0.5,
    label     = 'amostras de teste'
    )


plt.xlabel('X')
plt.ylabel('Y')

plt.legend()

plt.show()

#----------------------------------------------------------------------------
# TREINAR E TESTAR UM MODELO DE REGRESSÃO POLINOMIAL
#----------------------------------------------------------------------------

# instanciar e ajustar um objeto PolynomialFeatures

pf = PolynomialFeatures(degree=grau_polinomial)
pf = pf.fit(x_treino)

# transformar a matriz x incluindo os atributos polinomiais

x_treino_trans = pf.transform(x_treino)
x_teste_trans  = pf.transform(x_teste)
x_base_trans   = pf.transform(x_base)

# instanciar e treinar um modelo de regressão linear

modelo = LinearRegression()
modelo = modelo.fit(x_treino_trans,y_treino)

# obter respostas do modelo DENTRO e FORA da amostra

y_resposta_treino = modelo.predict(x_treino_trans)
y_resposta_teste  = modelo.predict(x_teste_trans)
y_resposta_base   = modelo.predict(x_base_trans)

# calcular métricas de erro das respostas

rmse_in  = math.sqrt(mean_squared_error(y_resposta_treino,y_treino))
rmse_out = math.sqrt(mean_squared_error(y_resposta_teste,y_teste))

#----------------------------------------------------------------------------
# VISUALIZAR GRAFICAMENTE OS RESULTADOS
#----------------------------------------------------------------------------

plt.figure(figsize=(16,7))

quadro1 = plt.subplot(121)
plt.ylim(-1.5,1.5)

quadro2 = plt.subplot(122)
plt.ylim(-1.5,1.5)

# exibir resultados DENTRO da amostra

quadro1.title.set_text(
    ( "Regressão de grau %d\n" % grau_polinomial ) +
    "Desempenho DENTRO da amostra\n" +
    ( "RMSE: %.4f" % rmse_in )
    )

quadro1.plot(
    x_base,
    y_base,
    color     = 'gray',
    linestyle = 'dotted',
    label     = 'Função-alvo (desconhecida)'
    )

quadro1.scatter(
    x_treino,
    y_treino,
    color     = 'green',
    marker    = 'o',
    s         = 30,
    alpha     = 0.5,
    label     = 'amostras de treinamento'
    )

quadro1.scatter(
    x_treino,
    y_resposta_treino,
    color     = 'blue',
    marker    = 'x',
    s         = 30,
    alpha     = 0.5,
    label     = 'respostas do modelo'
    )

quadro1.plot(
    x_base,
    y_resposta_base,
    color     = 'blue',
    linestyle = 'dotted',
    label     = 'função de decisão'
    )


# exibir resultados FORA da amostra

quadro2.title.set_text(
    ( "Regressão de grau %d\n" % grau_polinomial ) +
    "Desempenho FORA da amostra\n" +
    ( "RMSE: %.4f" % rmse_out )
    )

quadro2.plot(
    x_base,
    y_base,
    color     = 'gray',
    linestyle = 'dotted',
    label     = 'Função-alvo (desconhecida)'
    )

quadro2.scatter(
    x_teste,
    y_teste,
    color     = 'red',
    marker    = 'o',
    s         = 30,
    alpha     = 0.5,
    label     = 'amostras de teste'
    )

quadro2.scatter(
    x_teste,
    y_resposta_teste,
    color     = 'blue',
    marker    = 'x',
    s         = 30,
    alpha     = 0.5,
    label     = 'respostas do modelo'
    )

quadro2.plot(
    x_base,
    y_resposta_base,
    color     = 'blue',
    linestyle = 'dotted',
    label     = 'função de decisão'
    )

plt.show()


