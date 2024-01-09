import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Datos de entrada y salida
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output_data = np.array([0, 1, 1, 0])

# Crear una matriz de colores para visualizar las salidas
colors = ['red' if output == 0 else 'blue' for output in output_data]

# Visualización de datos de entrada en un gráfico de dispersión
plt.scatter(input_data[:, 0], input_data[:, 1], c=colors, marker='o')

# Títulos y etiquetas
plt.title('Visualización de la Operación XOR')
plt.xlabel('Entrada A')
plt.ylabel('Entrada B')

# Crear el modelo de red neuronal
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(input_data, output_data, epochs=1000, verbose=0)

# Hacer predicciones
predictions = model.predict(input_data)

# Visualizar las predicciones en el mismo gráfico
for i, prediction in enumerate(predictions):
    plt.annotate(f'{round(prediction[0], 2)}', (input_data[i, 0] + 0.1, input_data[i, 1]))

# Mostrar el gráfico
plt.show()
