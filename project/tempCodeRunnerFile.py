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
w_gd_0 = read_w_values('c:/Users/Mehmet Ali/Desktop/Department/2/optimization/project/results/w_gd_0.txt')
w_gd_1 = read_w_values('c:/Users/Mehmet Ali/Desktop/Department/2/optimization/project/results/w_gd_1.txt')
w_gd_2 = read_w_values('c:/Users/Mehmet Ali/Desktop/Department/2/optimization/project/results/w_gd_2.txt')
w_gd_3 = read_w_values('c:/Users/Mehmet Ali/Desktop/Department/2/optimization/project/results/w_gd_3.txt')
w_gd_4 = read_w_values('c:/Users/Mehmet Ali/Desktop/Department/2/optimization/project/results/w_gd_4.txt')

# 2. T-SNE ile 2D'ye indirgeme
tsne = TSNE(n_components=2, random_state=42)
w_gd_0_2d = tsne.fit_transform(w_gd_0)
w_gd_1_2d = tsne.fit_transform(w_gd_1)
w_gd_2_2d = tsne.fit_transform(w_gd_2)
w_gd_3_2d = tsne.fit_transform(w_gd_3)
w_gd_4_2d = tsne.fit_transform(w_gd_4)

# 3. 2D verileri görselleştirme
plt.figure(figsize=(10, 8))

# Gradient Descent 0'dan başladığı halinin çizimi
plt.plot(w_gd_0_2d[:, 0], w_gd_0_2d[:, 1], label="Gradient Descent 0", color='darkblue', marker='o', markersize=2)
for i in range(len(w_gd_0_2d)):
    if i % 10 == 0:
        plt.text(w_gd_0_2d[i, 0], w_gd_0_2d[i, 1], str(i), fontsize=8, ha='right', color='black')

# Gradient Descent 1'den başlayarak çizim
plt.plot(w_gd_1_2d[:, 0], w_gd_1_2d[:, 1], label="Gradient Descent 1", color='blue', marker='o', markersize=2)
for i in range(len(w_gd_1_2d)):
    if i % 10 == 0:
        plt.text(w_gd_1_2d[i, 0], w_gd_1_2d[i, 1], str(i), fontsize=8, ha='right', color='black')

# Gradient Descent 2'den başlayarak çizim
plt.plot(w_gd_2_2d[:, 0], w_gd_2_2d[:, 1], label="Gradient Descent 2", color='lightblue', marker='o', markersize=2)
for i in range(len(w_gd_2_2d)):
    if i % 10 == 0:
        plt.text(w_gd_2_2d[i, 0], w_gd_2_2d[i, 1], str(i), fontsize=8, ha='right', color='black')

# Gradient Descent 3'den başlayarak çizim
plt.plot(w_gd_3_2d[:, 0], w_gd_3_2d[:, 1], label="Gradient Descent 3", color='cyan', marker='o', markersize=2)
for i in range(len(w_gd_3_2d)):
    if i % 10 == 0:
        plt.text(w_gd_3_2d[i, 0], w_gd_3_2d[i, 1], str(i), fontsize=8, ha='right', color='black')

# Gradient Descent 4'den başlayarak çizim
plt.plot(w_gd_4_2d[:, 0], w_gd_4_2d[:, 1], label="Gradient Descent 4", color='darkcyan', marker='o', markersize=2)
for i in range(len(w_gd_4_2d)):
    if i % 10 == 0:
        plt.text(w_gd_4_2d[i, 0], w_gd_4_2d[i, 1], str(i), fontsize=8, ha='right', color='black')

# Grafiğin genel ayarları
plt.title('T-SNE Visualization of Gradient Descent Methods')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()
plt.grid(True)
plt.show()
