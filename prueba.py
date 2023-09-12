
from sklearn.model_selection import train_test_split

# Datos de ejemplo
datos = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Dividir los datos en conjuntos de entrenamiento y prueba sin especificar random_state
entrenamiento1, prueba1 = train_test_split(datos, test_size=0.9)

# Dividir los mismos datos en conjuntos de entrenamiento y prueba con random_state
entrenamiento2, prueba2 = train_test_split(datos, test_size=0.9, random_state=42)

print("División sin random_state:")
print("Entrenamiento1:", entrenamiento1)
print("Prueba1:", prueba1)

print("\nDivisión con random_state=42:")
print("Entrenamiento2:", entrenamiento2)
print("Prueba2:", prueba2)