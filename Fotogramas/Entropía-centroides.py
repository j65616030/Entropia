 import cv2
import numpy as np
from sklearn.cluster import KMeans

# Leer el video
cap = cv2.VideoCapture('video.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calcular la entropía
    entropy = np.zeros(gray.shape)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            entropy[i, j] = np.log2(gray[i, j] + 1)

    # Definir el número de centroides
    K = 5

    # Crear un objeto KMeans
    kmeans = KMeans(n_clusters=K)

    # Asignar la entropía a los centroides
    kmeans.fit(entropy.reshape(-1, 1))

    # Obtener los centroides
    centroids = kmeans.cluster_centers_

    # Asignar la entropía a los centroides más cercanos
    labels = kmeans.labels_
    for i in range(entropy.shape[0]):
        for j in range(entropy.shape[1]):
            entropy[i, j] = centroids[labels[i * entropy.shape[1] + j]]

    # Mostrar el resultado
    cv2.imshow('Entropy', entropy)
    cv2.waitKey(1)

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
