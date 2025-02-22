import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

caminho = r'C:\Users\PC2\Downloads\archive\winequalityN.csv'
df = pd.read_csv(caminho)

df = df[df['type'].str.lower() == 'white']
df['opinion'] = df['quality'].apply(lambda x: 0 if x <= 5 else 1)
df = df.drop(['quality', 'type'], axis=1)

X = df.drop('opinion', axis=1).fillna(df.drop('opinion', axis=1).mean())
y = df['opinion']

pipeline_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=1000, random_state=42))
])

model_dt = DecisionTreeClassifier(random_state=42)

pipeline_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True, random_state=42))
])

models = {
    'Regressão Logística': pipeline_lr,
    'Árvore de Decisão': model_dt,
    'SVM': pipeline_svm
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fpr_grid = np.linspace(0, 1, 100)

plt.figure(figsize=(10, 8))
model_auc_mean = {}

for model_name, model in models.items():
    tprs = []
    aucs = []
    for train_index, test_index in cv.split(X, y):
        X_cv_train, X_cv_val = X.iloc[train_index], X.iloc[test_index]
        y_cv_train, y_cv_val = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_cv_train, y_cv_train)
        # Probabilidades da classe positiva
        y_prob = model.predict_proba(X_cv_val)[:, 1]

        fpr, tpr, _ = roc_curve(y_cv_val, y_prob)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        interp_tpr = np.interp(fpr_grid, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(fpr_grid, mean_tpr)
    std_auc = np.std(aucs)
    model_auc_mean[model_name] = (mean_auc, std_auc)

    plt.plot(fpr_grid, mean_tpr, label=f"{model_name} (AUC = {mean_auc:.2f} ± {std_auc:.2f})")

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Aleatória')
plt.xlabel("Taxa de Falsos Positivos (FPR)")
plt.ylabel("Taxa de Verdadeiros Positivos (TPR)")
plt.title("Curva ROC Média dos Modelos")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

print("Comparação de Modelos (Média ± Desvio do AUC):")
for model_name, (mean_auc, std_auc) in model_auc_mean.items():
    print(f"{model_name}: AUC = {mean_auc:.2f} ± {std_auc:.2f}")
