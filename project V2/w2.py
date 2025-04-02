import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Fonksiyon: w parametrelerini bir dosyadan oku
def read_w_values(file_path):
    w_values = []
    with open(file_path, 'r') as file:
        for line in file:
            w_epoch = list(map(float, line.split()))
            w_values.append(w_epoch)
    return np.array(w_values)

# 1. Dosyalardan w değerlerini oku
w_gd = read_w_values('C:/Users/Mehmet Ali/Desktop/Department/2/optimization/project V2/results/w_gd.txt')
w_sgd = read_w_values('C:/Users/Mehmet Ali/Desktop/Department/2/optimization/project V2/results/w_sgd.txt')
w_adam = read_w_values('C:/Users/Mehmet Ali/Desktop/Department/2/optimization/project V2/results/w_adam.txt')

# 2. T-SNE ile 2D'ye indirgeme
tsne = TSNE(n_components=2, random_state=42)
w_gd_2d = tsne.fit_transform(w_gd)
w_sgd_2d = tsne.fit_transform(w_sgd)
w_adam_2d = tsne.fit_transform(w_adam)

# 3. 2D verileri görselleştirme
plt.figure(figsize=(10, 8))

# Gradient Descent çizimi
plt.plot(w_gd_2d[:, 0], w_gd_2d[:, 1], label="Gradient Descent", color='blue', marker='o', markersize=2)
for i in range(len(w_gd_2d)):
    if i % 10 == 0:
        plt.text(w_gd_2d[i, 0], w_gd_2d[i, 1], str(i), fontsize=8, ha='right', color='black')

# Stochastic Gradient Descent çizimi
plt.plot(w_sgd_2d[:, 0], w_sgd_2d[:, 1], label="SGD", color='green', marker='s', markersize=2)
for i in range(len(w_sgd_2d)):
    if i % 10 == 0:
        plt.text(w_sgd_2d[i, 0], w_sgd_2d[i, 1], str(i), fontsize=8, ha='right', color='black')

# Adam çizimi
plt.plot(w_adam_2d[:, 0], w_adam_2d[:, 1], label="Adam", color='red', marker='^', markersize=2)
for i in range(len(w_adam_2d)):
    if i % 10 == 0:
        plt.text(w_adam_2d[i, 0], w_adam_2d[i, 1], str(i), fontsize=8, ha='right', color='black')

# Grafiğin genel ayarları
plt.title('T-SNE Visualization of Optimization Methods')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()
plt.grid(True)
plt.show()
