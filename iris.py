#==============================================================================
# EXPERIMENTO 01 - Explorar e visualizar o conjunto de dados IRIS
#==============================================================================

#------------------------------------------------------------------------------
# Importar a biblioteca Pandas e PyPlot
#------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# Carregar o conjunto de dados IRIS do arquivo CSV
#------------------------------------------------------------------------------

dados = pd.read_csv('Iris_Data.csv', delimiter=',', decimal='.')


#------------------------------------------------------------------------------
# Exibir informações sobre o conjunto de dados
#------------------------------------------------------------------------------

print("\n\n Exibir as 4 primeiras amostras: \n")

print(dados.head(n=4))


print("\n\n Exibir as 3 últimas amostras: \n")

print(dados.tail(n=3))


print("\n\n Exibir as dimensões do conjunto de dados: \n")

print(dados.shape)
print("O conjunto tem",dados.shape[0],"amostras com",dados.shape[1],"variáveis")


print("\n\n Exibir os tipos das variáveis do conjunto de dados: \n")

print(dados.dtypes)


print("\n\n Exibir as 5 primeiras amostras de um dataframe somente com dados das pétalas: \n")

dados_das_petalas = dados[ ['petal_length','petal_width'] ]
print(dados_das_petalas.head())


print("\n\n Retirar o prefixo 'Iris-' do nome da espécie: \n")

dados['species'] = dados['species'].str.replace('Iris-','')
print(dados.head())

print("\n\n Outra forma (mais flexível) de retirar o prefixo 'Iris-': \n")

dados['species'] = dados['species'].apply(lambda r: r.replace('Iris-',''))
print(dados.head())

print("\n\n Contabilizar a quantidade de amostras de cada espécie: \n")

print(dados['species'].value_counts())

print("\n\n Exibir informações estatísticas sobre os dados: \n")

print(dados.describe())


print("\n\n Exibir a média de cada coluna: \n")

print(dados.mean())


print("\n\n Exibir a mediana de cada coluna: \n")
print(dados.median())


print("\n\n Exibir desvio-padrão de cada coluna: \n")
print(dados.std())

print("\n\n Exibir a média de cada coluna por espécie: \n")

print(dados.groupby('species').mean())

print("\n\n Exibir o desvio-padrão de cada coluna por espécie: \n")

print(dados.groupby('species').std())

print("\n\n Montar tabela com informações estatísticas personalizadas: \n")

resultado = dados.groupby('species').agg(
    {
     'petal_length': ['median','mean','std'],
     'petal_width' : ['median','mean','std']
     }   
    )

print(resultado)

print("\n\n Montar tabela com informações estatísticas de todas os atributos: \n")

resultado = dados.groupby('species').agg(
    {
     x: ['median','mean','std'] for x in dados.columns if x != 'species'
     }   
    )

print(resultado.to_string())

#------------------------------------------------------------------------------
# Exibir gráficos
#------------------------------------------------------------------------------

print("\n\n Visualizar o histograma de uma variável: \n")

grafico = dados['petal_length'].plot.hist(bins=30)

grafico.set(
    title  = 'DISTRIBUIÇÃO DO COMPRIMENTO DA PÉTALA',
    xlabel = 'Comprimento da Pétala (cm)',
    ylabel = 'Número de amostras'
    )

plt.show()


print("\n\n Visualizar o diagrama de dispersão entre duas variável: \n")

grafico = dados.plot.scatter('petal_width','petal_length')

grafico.set(
    title  = 'DISPERSÃO LARGURA vs COMPRIMENTO DA PÉTALA',
    xlabel = 'Largura da Pétala (cm)',
    ylabel = 'Comprimento da Pétala (cm)'
    )

plt.show()

#------------------------------------------------------------------------------
# Separar os atributos e o alvo em dataframes distintos
#------------------------------------------------------------------------------

atributos = dados.iloc[:,:4]
rotulos   = dados.iloc[:,4]

#------------------------------------------------------------------------------
# Fazer a mesma coisa com uso de índice negativo
#   - os atributos são todas as colunas exceto a última
#   - os rótulos estão na última coluna
#------------------------------------------------------------------------------

atributos = dados.iloc[:,:-1]
rotulos   = dados.iloc[:,-1]

#------------------------------------------------------------------------------
# Montar lista com os valores distintos dos rótulos (classes)
#------------------------------------------------------------------------------

classes = dados['species'].unique().tolist()

#------------------------------------------------------------------------------
# Montar mapa de cores associando cada classe a uma cor
#------------------------------------------------------------------------------

mapa_de_cores = ['red','green','blue']
cores_das_amostras = [ mapa_de_cores[classes.index(r)] for r in rotulos]

#------------------------------------------------------------------------------
# Visualizar a matriz de dispersão dos atributos
#------------------------------------------------------------------------------

pd.plotting.scatter_matrix(
    atributos,
    c=cores_das_amostras,
    figsize=(13,13),
    marker='o',
    s=50,
    alpha=0.5,
    diagonal='hist',         # 'hist' ou 'kde'
    hist_kwds={'bins':20}
    )

plt.suptitle(
    'MATRIZ DE DISPERSÃO DOS ATRIBUTOS',
    y=0.9,
    fontsize='xx-large'
    )

plt.show()

#------------------------------------------------------------------------------
# Visualizar um gráfico de dispersão 3D entre 3 atributos
#------------------------------------------------------------------------------

# escolher as variáveis de cada eixo

eixo_x = 'sepal_length'
eixo_y = 'petal_length'
eixo_z = 'petal_width'

# criar uma figura

figura = plt.figure(figsize=(15,12))

# criar um grafico 3D dentro da figura

grafico = figura.add_subplot(111,projection='3d')

# plotar o diagrama de dispersão 3D

grafico.scatter(
    dados[eixo_x],
    dados[eixo_y],
    dados[eixo_z],
    c=cores_das_amostras,
    marker='o',
    s=40,
    alpha=1.0
    )

plt.suptitle(
    'GRÁFICO DE DISPERSÃO 3D ENTRE 3 VARIÁVEIS',
    y=0.85,
    fontsize='xx-large'
    )

grafico.set_xlabel(eixo_x, fontsize='xx-large')
grafico.set_ylabel(eixo_y, fontsize='xx-large')
grafico.set_zlabel(eixo_z, fontsize='xx-large')

plt.show()