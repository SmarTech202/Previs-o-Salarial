import pandas as pd
from os import system
from time import sleep
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


ARQ = pd.read_csv('Arquive.csv')
X = ARQ.drop('SM100k', axis=1)
Y = ARQ.SM100k

LB = LabelEncoder()
DTC = DecisionTreeClassifier()

X['L_company'] = LB.fit_transform(X['company'])
X['L_job'] = LB.fit_transform(X['job'])
X['L_degree'] = LB.fit_transform(X['degree'])
X = X.drop(['company', 'job', 'degree'], axis=1)
DTC.fit(X, Y)

print('[Seu salário é maior que R$100.000?]\n')
sleep(0.5)
print('[0] Abc Pharma')
sleep(0.5)
print('[1] Facebook')
sleep(0.5)
print('[2] Google\n')
sleep(1)
COMPANY = int(input('[Número] - Qual dessas companias você trabalha?: '))

system('cls')

print('[Seu salário é maior que R$100.000?]\n')
sleep(0.5)
print('[0] Business Manager')
sleep(0.5)
print('[1] Computer Programmer')
sleep(0.5)
print('[2] Sales Executive\n')
sleep(1)
JOB = int(input('[Número] - Quais desses trabalhos é o seu?: '))

system('cls')

print('[Seu salário é maior que R$100.000?]\n')
sleep(0.5)
print('[0] Bachelor')
sleep(0.5)
print('[1] Master\n')
sleep(1)
DEGREE = int(input('[Número] - Qual o seu nivel de conhecimento??: '))

RESPONSE = DTC.predict([[COMPANY,JOB,DEGREE]])
system('cls')

print('[Seu salário é maior que R$100.000?]\n')
sleep(0.5)
print(f'Resposta[{RESPONSE[0]}] - Precisão[{accuracy_score(Y, DTC.predict(X))}]')