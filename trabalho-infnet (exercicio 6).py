import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, classification_report

caminho = r'C:\Users\PC2\Downloads\archive\winequalityN.csv'
df_all = pd.read_csv(caminho)

df_all['opinion'] = df_all['quality'].apply(lambda x: 0 if x <= 5 else 1)

df_all = df_all.drop(['quality', 'type'], axis=1)

X_all = df_all.drop('opinion', axis=1).fillna(df_all.drop('opinion', axis=1).mean())
y_all = df_all['opinion']

best_model = joblib.load('svm_model.pkl')

y_all_pred = best_model.predict(X_all)

num_ruins = np.sum(y_all_pred == 0)
num_bons = np.sum(y_all_pred == 1)

print("Inferência na base completa (todos os vinhos):")
print("Número de vinhos ruins (opinion == 0):", num_ruins)
print("Número de vinhos bons (opinion == 1):", num_bons)

cm_all = confusion_matrix(y_all, y_all_pred)
report_all = classification_report(y_all, y_all_pred)

print("\nMatriz de Confusão na base completa:")
print(cm_all)
print("\nRelatório de Classificação na base completa:")
print(report_all)
