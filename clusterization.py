import numpy as np
import matplotlib.pyplot as plt
import random
import math
import tkinter as tk
from tkinter import scrolledtext


# Класс для реализации карты Кохонена (SOM - Self-Organizing Map)
class KohonenSOM:
    def __init__(self, x, y, input_len, learning_rate=0.5, sigma=0.5):
        self.x = x  # количество узлов в строке
        self.y = y  # количество узлов в столбце
        self.input_len = input_len  # размерность входных данных
        self.learning_rate = learning_rate  # начальная скорость обучения
        self.sigma = sigma  # начальное значение радиуса соседства
        self.weights = np.random.random(
            (x, y, input_len)
        )  # инициализация весов случайными значениями

    # Вычисление евклидова расстояния между двумя векторами
    def _euclidean_distance(self, vec1, vec2):
        return np.linalg.norm(vec1 - vec2)

    # Нахождение нейрона-победителя (BMU - Best Matching Unit) для данного образца
    def _find_bmu(self, sample):
        bmu_idx = np.argmin(np.linalg.norm(self.weights - sample, axis=2))
        return np.unravel_index(bmu_idx, (self.x, self.y))

    # Обновление весов нейронов
    def _update_weights(self, sample, bmu, iter, max_iter):
        lr = self.learning_rate * np.exp(
            -iter / max_iter
        )  # уменьшение скорости обучения с течением времени
        if self.sigma > 0:
            sigma_log = np.log(self.sigma) if self.sigma > 1 else 1
            sig = self.sigma * np.exp(
                -iter / (max_iter / sigma_log)
            )  # уменьшение радиуса соседства с течением времени
        else:
            sig = self.sigma

        # Обновление весов для каждого нейрона в SOM
        for i in range(self.x):
            for j in range(self.y):
                w = self.weights[i, j, :]
                d = self._euclidean_distance(np.array([i, j]), np.array(bmu))
                influence = np.exp(-(d**2) / (2 * sig**2)) if sig > 0 else 0
                self.weights[i, j, :] += lr * influence * (sample - w)

    # Обучение SOM на данных
    def train(self, data, num_iterations):
        for iter in range(num_iterations):
            for sample in data:
                bmu = self._find_bmu(sample)
                self._update_weights(sample, bmu, iter, num_iterations)

    # Визуализация результатов работы SOM
    def visualize(self, data):
        points = []
        plt.figure(figsize=(10, 10))
        for i, sample in enumerate(data):
            bmu = self._find_bmu(sample)
            points.append((i, bmu))  # Добавляем ID точки и ее координаты BMU
            plt.text(
                bmu[0] + 0.5,
                bmu[1] + 0.5,
                str(i),
                color=plt.cm.rainbow(i / len(data)),
                fontdict={"weight": "bold", "size": 11},
            )
        plt.xlim([0, self.x])
        plt.ylim([0, self.y])
        plt.title("SOM - Visualization of Clustering")
        plt.show()
        return points


# Функция для вычисления евклидова расстояния между двумя точками
def euclidean_distance(p1, p2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(p1, p2)))


# Функция для назначения точек кластерам на основе ближайших центроидов
def assign_points_to_clusters(points, centroids):
    clusters = {i: [] for i in range(len(centroids))}
    for point in points:
        distances = [euclidean_distance(point[1], centroid) for centroid in centroids]
        closest_centroid = distances.index(min(distances))
        clusters[closest_centroid].append(point[0])  # Сохраняем только ID точки
    return clusters


# Функция для обновления центроидов кластеров
def update_centroids(clusters, points):
    new_centroids = []
    for cluster in clusters.values():
        cluster_points = [
            points[i][1] for i in cluster
        ]  # Получаем координаты BMU точек кластера
        new_centroid = [sum(dim) / len(cluster_points) for dim in zip(*cluster_points)]
        new_centroids.append(new_centroid)
    return new_centroids


# Функция реализации алгоритма K-средних (k-means) без ограничения на количество точек в кластере
def kmeans_no_limit(points, k, max_iterations=100):
    centroids = random.sample([p[1] for p in points], k)
    for _ in range(max_iterations):
        clusters = assign_points_to_clusters(points, centroids)
        new_centroids = update_centroids(clusters, points)
        if new_centroids == centroids:
            break
        centroids = new_centroids
    return clusters


# Функция для вычисления суммарного внутрикластерного расстояния
def compute_wcss(clusters, points):
    wcss = 0
    for cluster_points in clusters.values():
        if len(cluster_points) > 0:
            centroid = np.mean([points[i][1] for i in cluster_points], axis=0)
            wcss += sum(
                euclidean_distance(points[i][1], centroid) ** 2 for i in cluster_points
            )
    return wcss


# Функция для определения оптимального значения k
def determine_optimal_k(points, max_k):
    max_k = min(max_k, len(points))  # Убедимся, что max_k не превышает количество точек
    wcss_values = []
    for k in range(1, max_k + 1):
        clusters = kmeans_no_limit(points, k)
        wcss = compute_wcss(clusters, points)
        wcss_values.append(wcss)

    # Поиск "локтя" с использованием угла наклона
    angles = np.diff(np.diff(wcss_values))
    optimal_k = np.argmin(angles) + 2  # +2, чтобы компенсировать разницу в размерности
    return optimal_k


# Пример данных котлов (скорость, объем, цена)
data = [
    [2, 2, 5],
    [3, 3, 7],
    [250, 250, 60],
    [350, 350, 80],
    [40000, 40000, 900000],
    [45000, 45000, 1000000],
]

# Создание и инициализация SOM
som = KohonenSOM(x=10, y=10, input_len=3, learning_rate=0.5, sigma=1.0)

# Обучение SOM
som.train(data, num_iterations=100)

# Визуализация результатов и получение координат точек
points = som.visualize(data)
print("Coordinates of BMUs for each sample:", points)

# Определение оптимального значения k
max_k = 10  # Максимальное количество кластеров для проверки
optimal_k = determine_optimal_k(points, max_k)
print("Optimal number of clusters (k):", optimal_k)

# Кластеризация методом K-средних с оптимальным значением k
clusters_no_limit = kmeans_no_limit(points, optimal_k)


# Функция для отображения данных и кластеров в окне
def display_clusters(data, clusters_no_limit, k):
    window = tk.Tk()
    window.title("Clusters Visualization")

    data_label = tk.Label(window, text="Входные данные:")
    data_label.pack()

    data_text = scrolledtext.ScrolledText(window, width=40, height=10)
    data_text.pack()
    for i, sample in enumerate(data):
        data_text.insert(tk.END, f"Котёл {i+1}: {sample}\n")

    clusters_no_limit_label = tk.Label(
        window, text=f"Группировка по {k} кластерам (без ограничения):"
    )
    clusters_no_limit_label.pack()

    clusters_no_limit_text = scrolledtext.ScrolledText(window, width=40, height=10)
    clusters_no_limit_text.pack()
    for cluster_id, cluster_points in clusters_no_limit.items():
        clusters_no_limit_text.insert(
            tk.END, f"Кластер {cluster_id}: {cluster_points}\n"
        )

    window.mainloop()


# Вывод финальных кластеров в отдельное окно
display_clusters(data, clusters_no_limit, optimal_k)
