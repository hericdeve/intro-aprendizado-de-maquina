#==============================================================================
# REGRESSOR LINEAR vs KNN - CONJUNTO BOSTON
#==============================================================================

import pandas as pd
import math
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, SGDRegressor
from sklearn.metrics         import mean_squared_error, r2_score
from sklearn.neighbors       import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import seaborn as sns
# from pandas_profiling import ProfileReport
#import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# Ler as amostras da planilha Excel e gravar como dataframe Pandas
#------------------------------------------------------------------------------
norte = ['bairro_Arruda', 'bairro_Campo Grande', 'bairro_Encruzilhada', 'bairro_Ponto de Parada', 'bairro_Rosarinho', 'bairro_Torreao', 'bairro_Agua Fria', 'bairro_Cajueiro' ]
sul = ['bairro_Boa Viagem', 'bairro_Imbiribeira', 'bairro_Ipsep','bairro_Pina','bairro_Cohab']
centro = [ 'bairro_Recife', 'bairro_Sto Amaro', 'bairro_Boa Vista', 'bairro_Cabanga', 'bairro_Ilha do Leite', 'bairro_Paissandu', 'bairro_Sto Antonio', 'bairro_S Jose', 'bairro_Soledade', 'bairro_Coelhos'
]
noroeste = ['bairro_Aflitos', 'bairro_Apipucos', 'bairro_Casa Amarela', 'bairro_Casa Forte', 'bairro_Derby', 'bairro_Dois Irmaos', 'bairro_Espinheiro', 'bairro_Gracas', 'bairro_Jaqueira', 'bairro_Monteiro', 'bairro_Parnamirim', 'bairro_Poco', 'bairro_Santana',
'bairro_Guabiraba', 'bairro_Macaxeira' ]
sudeste = ['bairro_Afogados', 'bairro_Bongi' , 'bairro_Areias', 'bairro_Estancia' ,'bairro_Barro', 'bairro_Jd S Paulo', 'bairro_Sancho', 'bairro_Tejipio' ]
oeste = ['bairro_Cordeiro', 'bairro_Ilha do Retiro', 'bairro_Iputinga', 'bairro_Madalena', 'bairro_Prado', 'bairro_Torre', 'bairro_Zumbi',  'bairro_Engenho do Meio', 'bairro_Caxanga', 'bairro_Cid Universitaria', 'bairro_Varzea'
]

dados = pd.read_csv("conjunto_de_treinamento.csv")
dados_teste = pd.read_csv("conjunto_de_teste.csv")
# prof = ProfileReport(dados)
# prof.to_file(output_file='analise_inicial_dados.html')
pd.set_option("display.max_rows", None)
dados = dados.drop(
    [
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
        "tipo",
        # "bairro",
    ],
)

dados_teste = pd.get_dummies(
    dados_teste,
    columns=[
        "tipo",
        # "bairro",
    ],
)

# dados = dados.rename(
#    ...:     columns={
# for item in dados.columns:
#     if item not in dados_teste.columns and item != "preco":
#         dados_teste[item] = dados[item]
#         for k in dados_teste[item]:
#             k = 0
# for item in dados.columns:
#     for bairro in norte:
#         if item in bairro:
#             dados.columns= dados.columns.str.replace(item, 'norte')
#     for bairro in sul:
#         if item in bairro:
#             dados.columns= dados.columns.str.replace(item, 'sul')
#     for bairro in sudeste:
#         if item in bairro:
#             dados.columns= dados.columns.str.replace(item, 'sudeste')
#     for bairro in noroeste:
#         if item in bairro:
#             dados.columns= dados.columns.str.replace(item, 'noroeste')
#     for bairro in oeste:
#         if item in bairro:
#             dados.columns= dados.columns.str.replace(item, 'oeste')
#     for bairro in centro:
#         if item in bairro:
#             dados.columns= dados.columns.str.replace(item, 'centro')
      

# print(dados.columns)
# print()
# print(dados_teste.columns)
   
#------------------------------------------------------------------------------
# Transferir valores dos atributos e rótulos para arrays X e Y
#------------------------------------------------------------------------------
# dados_selecionados = dados['Id', 'quartos', 'suites', 'vagas', 'area_util', 'area_extra',
#        'churrasqueira', 'estacionamento', 'piscina', 'playground',
#        ...
#        'diferenciais_sauna', 'diferenciais_sauna e campo de futebol',
#        'diferenciais_sauna e copa', 'diferenciais_sauna e esquina',
#        'diferenciais_sauna e frente para o mar',
#        'diferenciais_sauna e playground',
#        'diferenciais_sauna e quadra poliesportiva',
#        'diferenciais_sauna e sala de ginastica',
#        'diferenciais_sauna e salao de festas', 'diferenciais_vestiario']
with pd.option_context("display.max_seq_items", None):
    print(dados.dtypes)
x = dados.iloc[:,1:-1].values
y = dados.iloc[:,-1].values



#------------------------------------------------------------------------------
# Particionar X e Y em conjunto de treinamento e conjunto de teste
#------------------------------------------------------------------------------

x_treino, x_teste, y_treino, y_teste = train_test_split(
    x,
    y,
    test_size = 0.3,
    random_state = 123   
    )
x_teste_alt = dados_teste.iloc[:,1:].values

# x_treino = x
# y_treino = y
# print(x_treino.shape())
# print(x.dtype())
print()
# print(x_teste.shape())






#------------------------------------------------------------------------------
# Ajustar a escala dos atributos
#------------------------------------------------------------------------------

escala = StandardScaler()
escala_y = StandardScaler()
escala = escala.fit(x_treino)

x_treino = escala.transform(x_treino)
# x_teste  = escala.transform(x_teste)

escala_y = escala_y.fit(y_treino.reshape(-1, 1))

y_treino = escala_y.transform(y_treino.reshape(-1, 1))
# y_teste = escala.transform(y_teste.reshape(-1, 1))

standard_deviation = np.std(y_treino)
distance_from_mean = abs(y_treino - np.mean(y_treino))
max_deviations = 2
not_outlier = distance_from_mean < max_deviations * standard_deviation
no_outliers = y_treino[not_outlier]

index_no_outlier = np.where(not_outlier == False)[0]


sns.boxplot(no_outliers)

print( 'caracteristicas com outliers: ',len(x_treino), ' precos com outliers:' ,len(y_treino))

print("Precos outliers normalizados")
print(y_treino[index_no_outlier])
print("Precos outliers reais")
print(escala_y.inverse_transform(y_treino[index_no_outlier]))


x_treino = np.delete(x_treino, index_no_outlier, axis=0)

y_treino = y_treino[not_outlier]


#------------------------------------------------------------------------------
# Treinar um regressor linear
#------------------------------------------------------------------------------

k = 1
pf = PolynomialFeatures(degree=k)
pf = pf.fit(x_treino)

x_treino = pf.transform(x_treino)
x_teste = pf.transform(x_teste)
# x_teste_alt = pf.transform(x_teste_alt)

na = x_treino.shape[1]

regressor_linear = LinearRegression()

regressor_linear = regressor_linear.fit(x_treino,y_treino)


y_resposta_treino = regressor_linear.predict(x_treino)
y_resposta_teste  = regressor_linear.predict(x_teste)
# y_resposta_teste_alt  = regressor_linear.predict(x_teste_alt)

#------------------------------------------------------------------------------
# Calcular as métricas e comparar os resultados
#------------------------------------------------------------------------------
# modelos_parametros = {
#     'svm': {
#         'model': svm.SVC(),
#         'params' : {
#             'gamma' : ['scale', 'auto'],
#             'C': [1,10,20, 30, 40],
#             'kernel': ['rbf','linear', 'poly', 'sigmoid']
            
#         }  
#     },
# }

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
with pd.option_context("display.max_seq_items", None):
    print(dados.columns)
    
# y_resposta_teste_alt = escala_y.inverse_transform(y_resposta_teste_alt.reshape(1, -1))

# dict_resultado = {'Id': dados_teste['Id'], 'preco': y_resposta_teste_alt.tolist()[0]}
# df_resultado = pd.DataFrame(data=dict_resultado)
# df_resultado.to_csv (f'predicao_02.csv', index = False, header=True)

#lasso

regressor_lasso = Lasso(alpha=125000,
                        max_iter=10000
                       )


regressor_lasso = regressor_lasso.fit(x_treino,y_treino)

y_resposta_treino = regressor_lasso.predict(x_treino)
y_resposta_teste  = regressor_lasso.predict(x_teste)
# y_resposta_teste_alt  = regressor_lasso.predict(x_teste_alt)

    
# y_resposta_teste_alt = escala_y.inverse_transform(y_resposta_teste_alt.reshape(1, -1))

# dict_resultado = {'Id': dados_teste['Id'], 'preco': y_resposta_teste_alt.tolist()[0]}
# df_resultado = pd.DataFrame(data=dict_resultado)
# df_resultado.to_csv (f'predicao_02_lasso.csv', index = False, header=True)

rmspe = np.sqrt(
                np.mean(
                    np.square(((y_treino - y_resposta_treino) / y_treino)),
                    axis=0
                )
            )
        

mse_in  = mean_squared_error(y_treino,y_resposta_treino)
rmse_in = math.sqrt(mse_in)
r2_in   = r2_score(y_treino,y_resposta_treino)

mse_out  = mean_squared_error(y_teste,y_resposta_teste)
rmse_out = math.sqrt(mse_out)
r2_out   = r2_score(y_teste,y_resposta_teste)

print(f'--------------------------------RIDGE-{k}------------------------------------')
print(' NA        RMSE_IN       R^2 IN       RMSE_OUT       R^2 OUT    RMSPE-saga')
print('%4d  %12.4f  %12.4f  %12.4f  %12.4f ' % ( na , rmse_in , r2_in, rmse_out,  r2_out))


#------------------------------------------------------------------------------
# Treinar e testar um regressor KNN para vários valores do parâmetros
#------------------------------------------------------------------------------

scores = cross_val_score(regressor_linear, x_treino, y_treino, cv=8, n_jobs=-1)

print(
    # "k = %2d" % k,
    "scores =",
    scores,
    "acurácia média = %6.1f" % (100 * sum(scores) / 8),
)
print(' ')
print(' REGRESSOR KNN:')
print(' ')

print('  K   DENTRO da amostra  FORA da amostra')
print(' ---  -----------------  ---------------')

for k in range(1,21):
    
    regressor_knn = KNeighborsRegressor(
        n_neighbors = k,
        weights     = 'uniform'     # 'uniform' ou 'distance'
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


