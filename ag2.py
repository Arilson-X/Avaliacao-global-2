import pandas as pd
from sklearn.linear_model import Perceptron


def tratadados(dfname):
    df_dados = pd.read_csv(dfname, sep=',')
    df_dados = df_dados.replace('o', -1)
    df_dados = df_dados.replace('b', 0)
    df_dados = df_dados.replace('x', +1)
    df_dados = df_dados.replace('negativo', -1)
    df_dados = df_dados.replace('positivo', +1)

    df_dados.to_csv("dados_tratados.csv")

tratadados()
