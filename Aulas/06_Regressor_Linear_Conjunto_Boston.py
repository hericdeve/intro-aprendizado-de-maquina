==============================================================================
# REGRESSOR LINEAR - CONJUNTO BOSTON
#==============================================================================

import pandas as pd
import math

from sklearn.model_selection import train_test_split
from sklearn.linear_model    import LinearRegression
from sklearn.metrics         import mean_squared_error, r2_score

import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# Ler as amostras da planilha Excel e gravar como dataframe Pandas
#------------------------------------------------------------------------------

dados = pd.read_excel("D02_Boston.xlsx")

#------------------------------------------------------------------------------
# Transferir valores dos atributos e rótulos para arrays X e Y
#------------------------------------------------------------------------------

x = dados.iloc[:,1:-1].to_numpy()
y = dados.iloc[:,-1].to_numpy()

#------------------------------------------------------------------------------
# Particionar X e Y em conjunto de treinamento e conjunto de teste
#------------------------------------------------------------------------------

x_treino, x_teste, y_treino, y_teste = train_test_split(
    x,
    y,
    test_size = 200,
    random_state = 0   
    )

#------------------------------------------------------------------------------
# Treinar um regressor linear
#------------------------------------------------------------------------------

regressor_linear = LinearRegression()

regressor_linear = regressor_linear.fit(x_treino,y_treino)

#------------------------------------------------------------------------------
# Obter as respostas do regressor linear dentro e fora da amostra
#------------------------------------------------------------------------------

y_resposta_treino = regressor_linear.predict(x_treino)
y_resposta_teste  = regressor_linear.predict(x_teste)

#------------------------------------------------------------------------------
# Calcular as métricas e comparar os resultados
#------------------------------------------------------------------------------

print(' ')
print(' REGRESSOR LINEAR:')
print(' ')

print(' Métrica  DENTRO da amostra  FORA da amostra')
print(' -------  -----------------  ---------------')

mse_in  = mean_squared_error(y_treino,y_resposta_treino)
rmse_in = math.sqrt(mse_in)
r2_in   = r2_score(y_treino,y_resposta_treino)

mse_out  = mean_squared_error(y_teste,y_resposta_teste)
rmse_out = math.sqrt(mse_out)
r2_out   = r2_score(y_teste,y_resposta_teste)

print(' %7s  %17.4f  %15.4f' % (  'mse' ,  mse_in ,  mse_out ) )
print(' %7s  %17.4f  %15.4f' % ( 'rmse' , rmse_in , rmse_out ) )
print(' %7s  %17.4f  %15.4f' % (   'r2' ,   r2_in ,   r2_out ) )

#------------------------------------------------------------------------------
# Plotar diagrama de dispersão entre a resposta correta e a resposta do modelo
#------------------------------------------------------------------------------

plt.scatter(x=y_teste,y=y_resposta_teste)
