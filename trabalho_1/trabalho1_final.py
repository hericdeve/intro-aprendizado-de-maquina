# -*- coding: utf-8 -*-

from numpy.core.numeric import NaN
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from pandas.core.frame import DataFrame
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

#Importar os dados

dados = pd.read_csv('conjunto_de_treinamento.csv')
dados_teste = pd.read_csv("conjunto_de_teste.csv")

# dados = pd.read_csv('conjunto_de_treinamento.csv')
# dados_teste = pd.read_csv("conjunto_de_teste.csv")


pd.set_option("display.max_rows", None)


dados = dados.drop(
    [
        "id_solicitante",
        "grau_instrucao",
        # "qtde_contas_bancarias_especiais",
        "codigo_area_telefone_trabalho",
        "estado_onde_trabalha",
        "codigo_area_telefone_residencial",
        "estado_onde_reside",
        "estado_onde_nasceu",
        # "local_onde_reside",
        "local_onde_trabalha",
        "possui_telefone_celular",
        'grau_instrucao_companheiro',
        'profissao_companheiro',
        
    ],
    axis=1,
)
dados_teste = dados_teste.drop(
    [
        # "id_solicitante",
        "grau_instrucao",
        # "qtde_contas_bancarias_especiais",
        "codigo_area_telefone_trabalho",
        "estado_onde_trabalha",
        "codigo_area_telefone_residencial",
        "estado_onde_reside",
        "estado_onde_nasceu",
        # "local_onde_reside",
        "local_onde_trabalha",
        "possui_telefone_celular",
        'grau_instrucao_companheiro',
        'profissao_companheiro',
    ],
    axis=1,
)


binarizador = LabelBinarizer()
for v in [
    "possui_telefone_residencial",
    "vinculo_formal_com_empresa",
    "possui_telefone_trabalho",
    "tipo_endereco",
]:
    dados[v] = binarizador.fit_transform(dados[v])

binarizador = LabelBinarizer()
for v in [
    "possui_telefone_residencial",
    "vinculo_formal_com_empresa",
    "possui_telefone_trabalho",
    "tipo_endereco",
]:
    dados_teste[v] = binarizador.fit_transform(dados_teste[v])



dados = pd.get_dummies(
    dados,
    columns=[
        # "produto_solicitado",
        "forma_envio_solicitacao",
        "sexo",
        # "dia_vencimento",
        # "estado_civil",
        # "nacionalidade",
        # "tipo_residencia",
        # "profissao",
        # "ocupacao",
        # "profissao_companheiro",
    ],
)
dados_teste = pd.get_dummies(
    dados_teste,
    columns=[
        # "produto_solicitado",
        "forma_envio_solicitacao",
        "sexo",
        # "dia_vencimento",
    #     "estado_civil",
    #     "nacionalidade",
    #     "tipo_residencia",
    #     "profissao",
    #     "ocupacao",
    #     "profissao_companheiro",
    ],
)



dados = dados.dropna()

#Preenche colunas com valores vazios com a moda


dados_teste['profissao'] = dados_teste['profissao'].fillna(dados_teste['profissao'].mode()[0])
dados_teste['ocupacao'] = dados_teste['ocupacao'].fillna(dados_teste['ocupacao'].mode()[0])
dados_teste['meses_na_residencia'] = dados_teste['meses_na_residencia'].fillna(dados_teste['meses_na_residencia'].mode()[0])


atributos_selecionados_teste = [
# 'sexo_ ',
'sexo_M',
'sexo_F',
'sexo_N',
'possui_email',
# 'local_onde_reside',
'tipo_endereco',
'idade',
'estado_civil',
'qtde_dependentes',
'dia_vencimento',
'possui_telefone_residencial',
'meses_na_residencia',
'renda_mensal_regular',
'renda_extra',
'possui_cartao_visa',
'possui_cartao_mastercard',
'possui_outros_cartoes',
'qtde_contas_bancarias',
# 'qtde_contas_bancarias_especiais',
'valor_patrimonio_pessoal',
'possui_carro',
'vinculo_formal_com_empresa',
'meses_no_trabalho',
'profissao',
'ocupacao',
'forma_envio_solicitacao_correio',
'forma_envio_solicitacao_internet',
'forma_envio_solicitacao_presencial',
]

atributos_selecionados = atributos_selecionados_teste + ["inadimplente"]

dados_selecinados = dados[atributos_selecionados]
dados_teste_selecionados = dados_teste[atributos_selecionados_teste]


# dados.plot.scatter(x='profissao',y='inadimplente')
dados_embaralhados = dados_selecinados.sample(frac=1, random_state=12345)
# dados_embaralhados_teste = dados_teste_selecionados.sample(frac=1, random_state=777)



x = dados_embaralhados.loc[:, dados_embaralhados.columns != "inadimplente"].values
y = dados_embaralhados.loc[:, dados_embaralhados.columns == "inadimplente"].values

x_teste = dados_teste_selecionados.loc[
    :, dados_teste_selecionados.columns != "inadimplente"
].values


x_treino = x
y_treino = y.ravel()


# x_treino, x_teste, y_treino, y_teste = train_test_split(
#     x, y.ravel(), train_size=19999, shuffle=True, random_state=12345
# )

ajustador_de_escala = MinMaxScaler()
# ajustador_de_escala_teste = MinMaxScaler()
ajustador_de_escala.fit(x_treino)
# ajustador_de_escala_teste.fit(x_teste)

x_treino = ajustador_de_escala.transform(x_treino)
x_teste = ajustador_de_escala.transform(x_teste)

#Aplicacao de modelos de classificacao

modelos_parametros = {
    
    'random_forest': {
        'model': RandomForestClassifier(random_state=12345),
        'params' : {
            'n_estimators': [100 ,200, 250, 300, 350, 400 ,500 ,700, 900, 1000],
            'max_depth' : [3, 4, 5, 6, 7, 8, 9, 10],
            'min_samples_split' : [2],
            'min_samples_leaf' : [1],
            'min_weight_fraction_leaf' : [0.0],
            'criterion' : ['entropy', 'gini'],
            

        }
    },

    'kneighbors_classifier':{
        'model': KNeighborsClassifier(),
        'params' : {
            'n_neighbors': [k for k in range(50, 300, 10)],
            'weights': ['uniform', 'distance'],
            'p' : [2, 1],
        
        }
    },
   

    'logistic_regression' : {
        'model': LogisticRegression(random_state=12345),
        'params': {
            'C': [pow(10,k) for k in range(-6,7)],
            'penalty': ['l1'],
            'solver' : ['liblinear'],
            'max_iter' : [10000],
          
           
        
        }
    }

}


print()
print('    Modelo              Best score')
print(" ---------------------------------------")
for model_name, mp in modelos_parametros.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(x_treino, y_treino)
    print(
        "%22s"%model_name,
        "%7.5f" % clf.best_score_
        )



#Criando classificador modelo Random Foreset

classificador = RandomForestClassifier(
        n_estimators=300, 
        random_state=12345,
        max_depth=6,              
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0
)



classificador = classificador.fit(x_treino, y_treino)


y_resposta_teste = classificador.predict(x_teste)

#criando arquivo respostas

d_teste = {'id_solicitante': dados_teste['id_solicitante'], 'inadimplente': y_resposta_teste.tolist()}
dados_resultado = pd.DataFrame(data=d_teste)
dados_resultado.to_csv ('prediction_teste_39.csv', index = False, header=True)



