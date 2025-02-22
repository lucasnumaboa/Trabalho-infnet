import pandas as pd

#QUestão 2
caminho = r'C:\Users\PC2\Downloads\archive\winequalityN.csv'
df = pd.read_csv(caminho)

df = df[df['type'].str.lower() == 'white']
df['opinion'] = df['quality'].apply(lambda x: 0 if x <= 5 else 1)
df = df.drop('quality', axis=1)

#questão 3
print("Variáveis presentes na base:")
print(df.columns.tolist())

print("\nTipos de variáveis (segundo pandas):")
print(df.dtypes)

def classify_column(col):
    if col.name == 'opinion':
        return 'categórica'
    if col.dtype == 'object':
        return 'categórica'
    elif pd.api.types.is_numeric_dtype(col):
        if pd.api.types.is_integer_dtype(col):
            return 'discreta' if col.nunique() < 15 else 'contínua'
        else:
            return 'contínua'
    else:
        return 'tipo não identificado'

print("\nClassificação automática dos tipos de variáveis:")
for col in df.columns:
    print(f"{col}: {classify_column(df[col])}")

print("\nMédias e Desvios Padrões:")
print(df.describe().loc[['mean', 'std']])
