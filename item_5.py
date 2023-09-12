import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Carga tus datos
data = pd.read_csv("./data/fashion-mnist_train.csv")
X = np.array(data.drop(['label'], axis=1))
y = np.array(data['label'])

# Definir los tipos de regularización
penalties = ['l0', 'l1', 'l2']

# Configura la validación cruzada de 5 particiones
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Inicializa una figura para trazar las matrices de confusión
fig, axes = plt.subplots(3, 5, figsize=(15, 9))

for penalty_idx, penalty in enumerate(penalties):
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Entrenar el modelo de regresión logística con la penalización seleccionada
        if penalty == 'l0':
            clf = LogisticRegression(penalty='none', max_iter=1000, random_state=42)
        else:
            clf = LogisticRegression(penalty=penalty, max_iter=1000, random_state=42)

        clf.fit(X_train, y_train)

        # Predecir en el conjunto de prueba
        y_pred = clf.predict(X_test)

        # Calcular la matriz de confusión
        cm = confusion_matrix(y_test, y_pred)

        # Dibuja la matriz de confusión en la sub-figura correspondiente
        ax = axes[penalty_idx, fold_idx]
        ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f'Penalty: {penalty}\nFold {fold_idx + 1}')
        tick_marks = np.arange(len(np.unique(y)))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)

        # Etiquetas de los ejes x e y
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

# Ajusta el espaciado entre sub-figuras
plt.tight_layout()

# Muestra la figura con las matrices de confusión
plt.show()
