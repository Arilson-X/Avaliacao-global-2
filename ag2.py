import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score


def verificaresultado(resultado):
    if resultado == 1:
        print('X ganhou')
    else:
        print('X perdeu')


def mostravelha(entrada):
    velha = []
    for j in range(0, 9):
        if entrada[j] == 1:
            velha.append('X')
        elif entrada[j] == 0:
            velha.append(' ')
        elif entrada[j] == -1:
            velha.append('O')
    print(f'|{velha[0]}|{velha[1]}|{velha[2]}| \n'
          f'|{velha[3]}|{velha[4]}|{velha[5]}| \n'
          f'|{velha[6]}|{velha[7]}|{velha[8]}|')


def tratadados(dfname):
    df_dados = pd.read_csv(dfname, sep=',')
    df_dados = df_dados.replace('o', -1)
    df_dados = df_dados.replace('b', 0)
    df_dados = df_dados.replace('x', +1)
    df_dados = df_dados.replace('negativo', -1)
    df_dados = df_dados.replace('positivo', +1)
    df_dados = df_dados.sort_index(ascending=False)

    df_dados.to_csv("dados_tratados.csv", index=False)
    print(df_dados)


tratadados('dados.csv')
df = pd.read_csv("dados_tratados.csv")
X = df.iloc[:, :9]

print(X)
Y = df.iloc[:, 9]
print(type(Y))
print(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)
classif = Perceptron(tol=1e-3, random_state=0).fit(x_train, y_train)

score = classif.score(x_test, y_test)
print(f'Score: {score}')
y_score = classif.predict(x_test)
accuracy = accuracy_score(y_test, y_score, normalize=False)
print(f'Accuracy: {accuracy} de {len(y_test)}')

entrada = []
for i in range(0, 9):
    entrada.append(int(input(f"Entre com o valor para a {i+1}ª posição: \n")))

mostravelha(entrada)

resultado = classif.predict([entrada])

print(verificaresultado(resultado[0]))




