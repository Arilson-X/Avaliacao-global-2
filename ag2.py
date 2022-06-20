import pandas as pd
import numpy as np
import matplotlib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron


def tratadados(dfname):
    df_dados = pd.read_csv(dfname, sep=',')
    df_dados = df_dados.replace('o', -1)
    df_dados = df_dados.replace('b', 0)
    df_dados = df_dados.replace('x', +1)
    df_dados = df_dados.replace('negativo', -1)
    df_dados = df_dados.replace('positivo', +1)

    df_dados.to_csv("dados_tratados.csv", index=False)
    print(df_dados)


tratadados('dados.csv')
df = pd.read_csv("dados_tratados.csv")
X = np.array(df.iloc[:, :9])
print(X.shape)
Y = np.array(df.iloc[:, 9].values)
print(Y.shape)

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=2)
classif = Perceptron(penalty='elasticnet').fit(x_train, y_train)
score = classif.score(x_test, y_test)
resultado = classif.predict([[1, -1, 1, 1, -1, -1, -1, 1, 1]])
print(score)
print(resultado)




