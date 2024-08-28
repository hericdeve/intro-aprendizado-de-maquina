#==============================================================================
# REGRESSOR LINEAR vs KNN vs POLINOMIAL - CONJUNTO BOSTON
#==============================================================================

import pandas as pd
import math

from sklearn.model_selection import train_test_split
from sklearn.linear_model    import LinearRegression
from sklearn.metrics         import mean_squared_error, r2_score
from sklearn.neighbors       import KNeighborsRegressor
from sklearn.preprocessing   import StandardScaler, PolynomialFeatures

from sklearn.linear_model    import Ridge, Lasso

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
# Ajustar a escala dos atributos
#------------------------------------------------------------------------------

escala = StandardScaler()

escala.fit(x_treino)

x_treino = escala.transform(x_treino)
x_teste  = escala.transform(x_teste)

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

#plt.scatter(x=y_teste,y=y_resposta_teste)

#------------------------------------------------------------------------------
# Treinar e testar um regressor KNN para vários valores do parâmetros
#------------------------------------------------------------------------------

print(' ')
print(' REGRESSOR KNN:')
print(' ')

print('  K   DENTRO da amostra  FORA da amostra')
print(' ---  -----------------  ---------------')

for k in range(1,21):
    
    regressor_knn = KNeighborsRegressor(
        n_neighbors = k,
        weights     = 'distance'     # 'uniform' ou 'distance'
        )

    regressor_knn = regressor_knn.fit(x_treino,y_treino)

    y_resposta_treino = regressor_knn.predict(x_treino)
    y_resposta_teste  = regressor_knn.predict(x_teste)

    mse_in  = mean_squared_error(y_treino,y_resposta_treino)
    rmse_in = math.sqrt(mse_in)
    r2_in   = r2_score(y_treino,y_resposta_treino)

    mse_out  = mean_squared_error(y_teste,y_resposta_teste)
    rmse_out = math.sqrt(mse_out)
    r2_out   = r2_score(y_teste,y_resposta_teste)

    print(' %3d  %17.4f  %15.4f' % ( k , rmse_in , rmse_out ) )

#------------------------------------------------------------------------------
# Treinar e testar um regressor POLINOMIAL para graus de 1 a 5
#------------------------------------------------------------------------------

print(' ')
print(' REGRESSOR POLINOMIAL DE GRAU K:')
print(' ')

print('  K    NA  DENTRO da amostra  FORA da amostra')
print(' ---  ---  -----------------  ---------------')

for k in range(1,6):
    
    pf = PolynomialFeatures(degree=k)
    
    pf = pf.fit(x_treino)
    x_treino_poly = pf.transform(x_treino)
    x_teste_poly = pf.transform(x_teste)
    
    regressor_linear = LinearRegression()
    
    regressor_linear = regressor_linear.fit(x_treino_poly,y_treino)
    
    y_resposta_treino = regressor_linear.predict(x_treino_poly)
    y_resposta_teste  = regressor_linear.predict(x_teste_poly)

    mse_in  = mean_squared_error(y_treino,y_resposta_treino)
    rmse_in = math.sqrt(mse_in)
    r2_in   = r2_score(y_treino,y_resposta_treino)

    mse_out  = mean_squared_error(y_teste,y_resposta_teste)
    rmse_out = math.sqrt(mse_out)
    r2_out   = r2_score(y_teste,y_resposta_teste)
    
    na = x_treino_poly.shape[1]

    print(' %3d  %4d  %17.4f  %15.4f' % ( k , na , rmse_in , rmse_out ) )

#------------------------------------------------------------------------------
# Treinar e testar um regressor RIDGE para graus de 1 a 5
#------------------------------------------------------------------------------

print(' ')
print(' REGRESSOR POLINOMIAL DE GRAU K COM REGULARIZACAO RIDGE (L2):')
print(' ')

print('  K    NA  DENTRO da amostra  FORA da amostra')
print(' ---  ---  -----------------  ---------------')

for k in range(1,6):
    
    pf = PolynomialFeatures(degree=k)
    
    pf = pf.fit(x_treino)
    x_treino_poly = pf.transform(x_treino)
    x_teste_poly = pf.transform(x_teste)
    
    regressor_ridge = Ridge(alpha=50.0)
    
    regressor_ridge = regressor_ridge.fit(x_treino_poly,y_treino)
    
    y_resposta_treino = regressor_ridge.predict(x_treino_poly)
    y_resposta_teste  = regressor_ridge.predict(x_teste_poly)

    mse_in  = mean_squared_error(y_treino,y_resposta_treino)
    rmse_in = math.sqrt(mse_in)
    r2_in   = r2_score(y_treino,y_resposta_treino)

    mse_out  = mean_squared_error(y_teste,y_resposta_teste)
    rmse_out = math.sqrt(mse_out)
    r2_out   = r2_score(y_teste,y_resposta_teste)
    
    na = x_treino_poly.shape[1]

    print(' %3d  %4d  %17.4f  %15.4f' % ( k , na , rmse_in , rmse_out ) )


print(' ')
print(' REGRESSOR POLINOMIAL DE GRAU K COM REGULARIZACAO LASSO (L1):')
print(' ')

print('  K    NA  DENTRO da amostra  FORA da amostra')
print(' ---  ---  -----------------  ---------------')

for k in range(1,5):
    
    pf = PolynomialFeatures(degree=k)
    
    pf = pf.fit(x_treino)
    x_treino_poly = pf.transform(x_treino)
    x_teste_poly = pf.transform(x_teste)
    
    regressor_lasso = Lasso(alpha=0.10,max_iter=100000)
    
    regressor_lasso = regressor_lasso.fit(x_treino_poly,y_treino)
    
    y_resposta_treino = regressor_lasso.predict(x_treino_poly)
    y_resposta_teste  = regressor_lasso.predict(x_teste_poly)

    mse_in  = mean_squared_error(y_treino,y_resposta_treino)
    rmse_in = math.sqrt(mse_in)
    r2_in   = r2_score(y_treino,y_resposta_treino)

    mse_out  = mean_squared_error(y_teste,y_resposta_teste)
    rmse_out = math.sqrt(mse_out)
    r2_out   = r2_score(y_teste,y_resposta_teste)
    
    na = x_treino_poly.shape[1]

    print(' %3d  %4d  %17.4f  %15.4f' % ( k , na , rmse_in , rmse_out ) )




  
