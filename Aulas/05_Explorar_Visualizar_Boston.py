#==============================================================================
# EXPERIMENTO 05 - EXPLORANDO E VISUALIZANDO O CONJUNTO "BOSTON" (REGRESSÃO)
#==============================================================================

import pandas as pd

from scipy.stats import pearsonr

#------------------------------------------------------------------------------
# Importar conjunto de dados de planilha Excel para dataframe Pandas
#------------------------------------------------------------------------------

dados = pd.read_excel("D02_Boston.xlsx")

#------------------------------------------------------------------------------
# Descartar a primeira coluna
#------------------------------------------------------------------------------

dados = dados.iloc[:,1:]

#------------------------------------------------------------------------------
# Verificar as colunas disponíveis
#------------------------------------------------------------------------------

colunas = dados.columns

print("Colunas disponíveis:")
print(colunas)

#------------------------------------------------------------------------------
# Plotar diagramas de dispersão entre cada atributo e o alvo
#------------------------------------------------------------------------------

for col in colunas:
    dados.plot.scatter(x=col,y='target')

#------------------------------------------------------------------------------
# Listar os coeficientes de Pearson entre cada atributo e o alvo
#------------------------------------------------------------------------------

for col in colunas:
    print('%10s = %6.3f' % ( col , pearsonr(dados[col],dados['target'])[0] ) )

#------------------------------------------------------------------------------
# Explorar correlações mútuas entre os atributos
#------------------------------------------------------------------------------

atributo1 = "LSTAT"
atributo2 = "PTRATIO"

print('%10s = %6.3f' % (
    atributo1+'_'+atributo2,
    pearsonr(dados[atributo1],dados[atributo2])[0] ) )

dados.plot.scatter(x=atributo1,y=atributo2)


