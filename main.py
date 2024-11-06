import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rc("figure", figsize=(10, 6))

# Carregamento dos dados
data = pd.read_csv('dataset.csv', header=None, names=['preco', 'tipo', 'area', 'quartos', 'bairro'], skiprows=1)
print(data.head())
X = data[['tipo', 'area', 'quartos', 'bairro']]
y = data['preco']

# Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criação do pipeline
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()), 
    ('regressor', RandomForestRegressor())  # Da pra tentar com RandomForestRegressor() ou LinearRegression()
])

# Validação cruzada para avaliação do modelo
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"MSE Médio com Validação Cruzada: {-np.mean(cv_scores)}")

# Treinamento do modelo
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Cálculo das métricas
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Erro Quadrático Médio: {rmse}")
print(f"Erro Absoluto Médio: {mae}")
print(f"Coeficiente de Determinação: {r2}")

# Gráfico
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel("Preço Real")
plt.ylabel("Preço Predito")
plt.title("Comparação entre Preço Real e Predito Com Os Dados Testes")
plt.xlim(0, 10000)
plt.ylim(0, 10000)
plt.show()

# Informações
print("\nPor favor, insira as informações do imóvel para previsão do preço conforme tabelas:")
print("\nApartamento: 1, \nKitnet: 2, \nHotel-Flat: 3, \nCasa: 4 \nComercial: 5")
print(
    "'Asa Norte': 1, 'Sudoeste': 2, 'Asa Sul': 3, 'Águas Claras': 4, 'Taguatinga': 5, "
    "'Lago Norte': 6, 'Lago Sul': 7, 'Guará II': 8, 'Samambaia': 9, 'Ceilândia': 10, "
    "'Núcleo Bandeirante': 11, 'Jardim Botânico': 12, 'Noroeste': 13, 'Guará I': 14, "
    "'Riacho Fundo': 15, 'Cruzeiro': 16, 'Park Sul': 17, 'Sobradinho': 18, 'Areal': 19, "
    "'Vicente Pires': 20, 'Mangueiral': 21, 'Park Way': 22, 'Gama': 23, 'Octogonal': 24, "
    "'Santa Maria': 25, 'Lúcio Costa': 26, 'Paranoá': 27, 'Recanto das Emas': 28, "
    "'Candangolândia': 29"
)

# Dados do cliente
tipo = int(input("Tipo do imóvel (exemplo: 1 = Apartamento, 2 = Casa, etc.): "))
area = float(input("Área do imóvel em m²: "))
quartos = int(input("Número de quartos: "))
bairro = int(input("Código do bairro (conforme tabela): "))
dados_usuario = pd.DataFrame([[tipo, area, quartos, bairro]], columns=['tipo', 'area', 'quartos', 'bairro'])

# Realiza a previsão com o modelo treinado
preco_previsto = pipeline.predict(dados_usuario)[0]
print(f"\nPreço estimado para o imóvel: R${preco_previsto:.2f}")
