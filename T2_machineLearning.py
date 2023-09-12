import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

data = pd.read_csv("./data/fashion-mnist_train.csv")

x = np.array(data.drop(['label'],axis=1))
y = np.array(data['label'])


# entrenamiento, prueba, clase_entrenamiento, clase_prueba
# El elegido es 60% entrenamiento, 20% validacion, 20% testeo
entrena_temp , test , entrena_temp_clase , test_clase = train_test_split(x, y, test_size=0.2, random_state=42)
entrenado , valida , entrenado_clase , valida_clase = train_test_split(entrena_temp, entrena_temp_clase, test_size=0.25, random_state=42)

""" 
    print("entrena_temp.shape[0]: " , entrena_temp.shape[0] )
    print("test.shape[0]: " , test.shape[0] )
    print("valor_objetivo.shape[0]: " , entrena_temp_clase.shape[0] )
    print("clase_datos_test.shape[0]: " , test_clase.shape[0] )

    #------------------------

    print("entrenado.shape[0]: " , entrenado.shape[0] )
    print("valida.shape[0]: " , valida.shape[0] )
    print("entrenado_clase.shape[0]: " , entrenado_clase.shape[0] )
    print("valida_clase.shape[0]: " , valida_clase.shape[0] ) 
"""

# Entrenamiento

# L0
""" clf = LogisticRegression(penalty='none', max_iter=1000, random_state=42)
clf.fit(entrenado, entrenado_clase) """

# L1
""" clf = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, random_state=42)
clf.fit(entrenado, entrenado_clase) """

# L2
clf = LogisticRegression(penalty='l2', max_iter=1000, random_state=42)
clf.fit(entrenado, entrenado_clase)

# Prediccion
pred_entrenamiento = clf.predict(entrenado)
pred_validacion = clf.predict(valida)
pred_testeo = clf.predict(test)

# Matrices
matriz_entrenamiento = confusion_matrix(entrenado_clase, pred_entrenamiento)
matriz_validacion = confusion_matrix(valida_clase, pred_validacion)
matriz_testeo = confusion_matrix(test_clase, pred_testeo)

# Imprimir las matrices de confusión
print("Matriz de Entrenamiento:")
print(matriz_entrenamiento)

print("\nMatriz de Validación:")
print(matriz_validacion)

print("\nMatriz de Prueba:")
print(matriz_testeo)