import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

caminho = r'C:\Users\PC2\Downloads\archive\winequalityN.csv'
df = pd.read_csv(caminho)

df = df[df['type'].str.lower() == 'white']

df['opinion'] = df['quality'].apply(lambda x: 0 if x <= 5 else 1)

df = df.drop(['quality', 'type'], axis=1)

X = df.drop('opinion', axis=1).fillna(df.drop('opinion', axis=1).mean())
y = df['opinion']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=1000, random_state=42))
])

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

cv_acc = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='accuracy')
cv_precision = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='precision_weighted')
cv_recall = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='recall_weighted')
cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='f1_weighted')

print("Validação Cruzada (10-fold):")
print("Acurácia: Média: {:.4f} | Desvio: {:.4f}".format(np.mean(cv_acc), np.std(cv_acc)))
print("Precisão: Média: {:.4f} | Desvio: {:.4f}".format(np.mean(cv_precision), np.std(cv_precision)))
print("Recall: Média: {:.4f} | Desvio: {:.4f}".format(np.mean(cv_recall), np.std(cv_recall)))
print("F1-score: Média: {:.4f} | Desvio: {:.4f}".format(np.mean(cv_f1), np.std(cv_f1)))

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, 'svm_model.pkl')

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nDesempenho no Conjunto de Teste:")
print("Acurácia:", accuracy)
print("\nMatriz de Confusão:")
print(cm)
print("\nRelatório de Classificação:")
print(report)
