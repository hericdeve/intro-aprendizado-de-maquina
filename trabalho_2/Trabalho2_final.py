#==============================================================================
# REGRESSOR LINEAR vs KNN vs POLINOMIAL - CONJUNTO BOSTON
#==============================================================================

import pandas as pd
import math
import numpy as np
from sklearn.base import BaseEstimator

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, SGDRegressor
from sklearn.metrics         import mean_squared_error, r2_score
from sklearn.neighbors       import KNeighborsRegressor
from sklearn.preprocessing   import StandardScaler, PolynomialFeatures
import seaborn as sns


#------------------------------------------------------------------------------
# Ler as amostras da planilha Excel e gravar como dataframe Pandas
#------------------------------------------------------------------------------
dados = pd.read_csv("conjunto_de_treinamento.csv")
dados_teste = pd.read_csv("conjunto_de_teste.csv")

pd.set_option("display.max_rows", None)
dados = dados.drop(
    [
        'Id',
        'tipo',
        "diferenciais",
        'tipo_vendedor',
        'bairro',
        'area_extra',
        'estacionamento',
        'churrasqueira',
        'piscina',
        'playground',
        'quadra',
        's_festas',
        's_jogos',
        's_ginastica',
        'sauna',
        'vista_mar'
    ],
    axis=1,
)

dados_teste = dados_teste.drop(
    [
        'tipo',
        "diferenciais",
        'tipo_vendedor',
        'bairro',
        'area_extra',
        'estacionamento',
        'churrasqueira',
        'piscina',
        'playground',
        'quadra',
        's_festas',
        's_jogos',
        's_ginastica',
        'sauna',
        'vista_mar'
    ],
    axis=1,
)

dados = pd.get_dummies(
    dados,
    columns=[
        # "tipo",
        # "bairro",
    ],
)

dados_teste = pd.get_dummies(
    dados_teste,
    columns=[
        # "tipo",
        # "bairro",
    ],
)

#------------------------------------------------------------------------------
# Transferir valores dos atributos e rótulos para arrays X e Y
#------------------------------------------------------------------------------

x = dados.iloc[:,:-1].values
y = dados.iloc[:,-1].values
x_teste_alt = dados_teste.iloc[:,1:].values


#------------------------------------------------------------------------------
# Particionar X e Y em conjunto de treinamento e conjunto de teste
#------------------------------------------------------------------------------


x_treino, x_teste, y_treino, y_teste = train_test_split(
    x,
    y,
    test_size = 0.3,
    random_state = 3  
    )



#------------------------------------------------------------------------------
# Ajustar a escala dos atributos e remover outliers
#------------------------------------------------------------------------------

escala = StandardScaler()
escala_y = StandardScaler()
escala = escala.fit(x_treino)

x_treino = escala.transform(x_treino)
x_teste  = escala.transform(x_teste)
x_teste_alt  = escala.transform(x_teste_alt)


escala_y = escala_y.fit(y_treino.reshape(-1, 1))

y_treino = escala_y.transform(y_treino.reshape(-1, 1))
y_teste = escala_y.transform(y_teste.reshape(-1, 1))

std = np.std(y_treino)
distancia = abs(y_treino - np.mean(y_treino))
criterio = distancia < 2 * std
no_outlier = y_treino[criterio]
index = np.where(criterio == False)[0]


sns.boxplot(no_outlier)


x_treino = np.delete(x_treino, index, axis=0)

y_treino = y_treino[criterio]

pf = PolynomialFeatures(degree=1)

pf = pf.fit(x_treino)
x_treino_poly = pf.transform(x_treino)
x_teste_poly = pf.transform(x_teste)

modelos_parametros = {
    # 'LinearRegression': {
    #     'model': LinearRegression(n_jobs=-1),
    #     'params' : {            
    #     }  
    # },
    'KNeighborsRegressor': {
        'model': KNeighborsRegressor(n_jobs=-1),
        'params' : {
            'n_neighbors' : [k for k in range(1, 300, 1)],
            'weights': ['uniform', 'distance'],
            'p': [1,2]
        }
   
    },
    # 'Ridge': {
    #     'model': Ridge(random_state=12345),
    #     'params' : {
    #         'alpha' : [k for k in range(1, 300000, 10000)],
          
    #     }
   
    # },
    # 'Lasso': {
    #     'model': Lasso(random_state=12345),
    #     'params' : {
    #         'alpha' : [k for k in range(1, 300000, 10000)],
    #         'max_iter': [10000],
      

    #     }
    # },
    # 'ElasticNet': {
    #     'model': ElasticNet(random_state=12345),
    #     'params' : {
            # 'alpha' : [k for k in range(1, 300000, 10000)],
    #         'l1_ratio' : [pow(10,-k/10) for k in range(30)],
    #         'normalize' : [False],
         

    #     }
    # },
    # 'SGDRegressor': {
    #     'model': SGDRegressor(random_state=12345),
    #     'params' : {
    #         # 'alpha' : [125000],
    #         'max_iter': [100],
    #         'tol' : [1e-3],
          
    #     }
    # },
    
}



# print()
# print('    Modelo              Best score(mse)')
# print(" ---------------------------------------")
# for model_name, mp in modelos_parametros.items():

#     clf =  GridSearchCV(mp['model'], mp['params'], cv=8, return_train_score=False, scoring=['neg_root_mean_squared_error', 'r2'], refit='neg_root_mean_squared_error', n_jobs=-1)
#     clf.fit(x_treino_poly, y_treino)
#     print(
#         "%22s"%model_name,
#         "%7.5f" % abs(clf.best_score_)
#         )

# print()


#------------------------------------------------------------------------------
# Treinar e testar um regressor KNN para vários valores do parâmetros
#------------------------------------------------------------------------------

print(' ')
print(' REGRESSOR KNN:')
print(' ')

print('  K   DENTRO da amostra  FORA da amostra')
print(' ---  -----------------  ---------------')

for k in range(25,35):
    
    regressor_knn = KNeighborsRegressor(
        n_neighbors = k, 
        weights     = 'distance',  
        p=1,
        n_jobs=-1
        )

    regressor_knn = regressor_knn.fit(x_treino,y_treino)

    y_resposta_treino = regressor_knn.predict(x_treino)
    y_resposta_teste  = regressor_knn.predict(x_teste)
    y_resposta_teste_alt = regressor_knn.predict(x_teste_alt)

    mse_in  = mean_squared_error(y_treino,y_resposta_treino)
    rmse_in = math.sqrt(mse_in)
    r2_in   = r2_score(y_treino,y_resposta_treino)

    mse_out  = mean_squared_error(y_teste,y_resposta_teste)
    rmse_out = math.sqrt(mse_out)
    r2_out   = r2_score(y_teste,y_resposta_teste)

    print(' %3d  %17.4f  %15.4f' % ( k , rmse_in , rmse_out ) )

regressor_knn = KNeighborsRegressor(
        n_neighbors = 31, 
        weights     = 'distance',
        p=1,
        n_jobs=-1
        )

regressor_knn = regressor_knn.fit(x_treino,y_treino)

y_resposta_teste_alt = regressor_knn.predict(x_teste_alt)

y_resposta_teste_alt = escala_y.inverse_transform(y_resposta_teste_alt).reshape(1, -1)

d_teste = {'Id': dados_teste['Id'], 'preco': y_resposta_teste_alt.tolist()[0]}
dados_resultado = pd.DataFrame(data=d_teste)
dados_resultado.to_csv ('regressao_knn_10.csv', index = False, header=True)


#------------------------------------------------------------------------------
# Treinar e testar um regressor POLINOMIAL para graus de 1 a 7
#------------------------------------------------------------------------------

print(' ')
print(' REGRESSOR POLINOMIAL DE GRAU K:')
print(' ')

print('  K    NA  DENTRO da amostra  FORA da amostra')
print(' ---  ---  -----------------  ---------------')

for k in range(1,8):
    
    pf = PolynomialFeatures(degree=k)
    
    pf = pf.fit(x_treino)
    x_treino_poly = pf.transform(x_treino)
    x_teste_poly = pf.transform(x_teste)
    
    regressor_linear = LinearRegression(n_jobs=-1)
    
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

y_resposta_teste_alt = escala_y.inverse_transform(y_resposta_teste_alt).reshape(1, -1)

# d_teste = {'Id': dados_teste['Id'], 'preco': y_resposta_teste_alt.tolist()[0]}
# dados_resultado = pd.DataFrame(data=d_teste)
# dados_resultado.to_csv ('regressao_polinomial_03.csv', index = False, header=True)


#------------------------------------------------------------------------------
# Treinar e testar um regressor RIDGE para graus de 1 a 7
#------------------------------------------------------------------------------

print(' ')
print(' REGRESSOR POLINOMIAL DE GRAU K COM REGULARIZACAO RIDGE (L2):')
print(' ')

print('  K    NA  DENTRO da amostra  FORA da amostra')
print(' ---  ---  -----------------  ---------------')
alp = 100000
for k in range(1,8):
    
    pf = PolynomialFeatures(degree=k)
    
    pf = pf.fit(x_treino)
    x_treino_poly = pf.transform(x_treino)
    x_teste_poly = pf.transform(x_teste)
    
    regressor_ridge = Ridge(alpha=alp)
    
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

    y_resposta_teste_alt = escala_y.inverse_transform(y_resposta_teste_alt).reshape(1, -1)

# d_teste = {'Id': dados_teste['Id'], 'preco': y_resposta_teste_alt.tolist()[0]}
# dados_resultado = pd.DataFrame(data=d_teste)
# dados_resultado.to_csv ('regressao_ridge_03.csv', index = False, header=True)


print(' ')
print(' REGRESSOR POLINOMIAL DE GRAU K COM REGULARIZACAO LASSO (L1):')
print(' ')

print('  K    NA  DENTRO da amostra  FORA da amostra')
print(' ---  ---  -----------------  ---------------')

for k in range(1,8):
    
    pf = PolynomialFeatures(degree=k)
    
    pf = pf.fit(x_treino)
    x_treino_poly = pf.transform(x_treino)
    x_teste_poly = pf.transform(x_teste)
    
    regressor_lasso = Lasso(alpha=alp,max_iter=10000)
    
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

y_resposta_teste_alt = escala_y.inverse_transform(y_resposta_teste_alt).reshape(1, -1)

# d_teste = {'Id': dados_teste['Id'], 'preco': y_resposta_teste_alt.tolist()[0]}
# dados_resultado = pd.DataFrame(data=d_teste)
# dados_resultado.to_csv ('regressao_lasso_04.csv', index = False, header=True)


print(' ')
print(' REGRESSOR POLINOMIAL DE GRAU K COM ElasticNet:')
print(' ')

print('  K    NA  DENTRO da amostra  FORA da amostra')
print(' ---  ---  -----------------  ---------------')

for k in range(1,8):
    
    pf = PolynomialFeatures(degree=k)
    
    pf = pf.fit(x_treino)
    x_treino_poly = pf.transform(x_treino)
    x_teste_poly = pf.transform(x_teste)
    
    regressor_elasticnet = ElasticNet(alpha=alp,l1_ratio=0.4, normalize=False)
    
    regressor_elasticnet = regressor_elasticnet.fit(x_treino_poly,y_treino)
    
    y_resposta_treino = regressor_elasticnet.predict(x_treino_poly)
    y_resposta_teste  = regressor_elasticnet.predict(x_teste_poly)

    mse_in  = mean_squared_error(y_treino,y_resposta_treino)
    rmse_in = math.sqrt(mse_in)
    r2_in   = r2_score(y_treino,y_resposta_treino)

    mse_out  = mean_squared_error(y_teste,y_resposta_teste)
    rmse_out = math.sqrt(mse_out)
    r2_out   = r2_score(y_teste,y_resposta_teste)
    
    na = x_treino_poly.shape[1]

    print(' %3d  %4d  %17.4f  %15.4f' % ( k , na , rmse_in , rmse_out ) )
y_resposta_teste_alt = escala_y.inverse_transform(y_resposta_teste_alt).reshape(1, -1)

# d_teste = {'Id': dados_teste['Id'], 'preco': y_resposta_teste_alt.tolist()[0]}
# dados_resultado = pd.DataFrame(data=d_teste)
# dados_resultado.to_csv ('regressao_elasticnet_03.csv', index = False, header=True)


#------------------------------------------------------------------------------
# Treinar e testar um regressor SGD com funcao de perda 'squared_loss'
#------------------------------------------------------------------------------

print(' ')
print(' REGRESSOR SGD:')
print(' ')

regressor_sgd = SGDRegressor(
    loss='squared_loss',
    alpha=0,
    penalty='l2',
    tol=1e-5,
    max_iter=100000
    )
regressor_sgd = regressor_sgd.fit(x_treino,y_treino)

y_resposta_treino = regressor_sgd.predict(x_treino)
y_resposta_teste  = regressor_sgd.predict(x_teste)

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

y_resposta_teste_alt = escala_y.inverse_transform(y_resposta_teste_alt).reshape(1, -1)

# d_teste = {'Id': dados_teste['Id'], 'preco': y_resposta_teste_alt.tolist()[0]}
# dados_resultado = pd.DataFrame(data=d_teste)
# dados_resultado.to_csv ('regressao_sgd_03.csv', index = False, header=True)

  
