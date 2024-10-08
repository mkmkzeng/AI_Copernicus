import os
import json
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import torch

# 文件夹路径
input_folder = './dataset/spectrograms'  # 输入文件夹，包含梅尔频谱图（.npy 格式）
output_json = './dataset/spectrogram_clusters.json'  # 输出：保存梅尔频谱图与类别号映射的 JSON 文件
NUM_CLUSTERS = 5  # 假设有5个类别

# 检查是否可以使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 提取频谱图数据
def load_spectrogram(spectrogram_file):
    """加载梅尔频谱图数据"""
    spectrogram = np.load(spectrogram_file)
    return torch.tensor(spectrogram, device=device)  # 将数据加载到 GPU 上（如果有 GPU）

# 聚类并保存结果
def cluster_spectrograms():
    spectrogram_files = [f for f in os.listdir(input_folder) if f.endswith('.npy')]
    features = []
    file_names = []

    print("Loading spectrograms and extracting features...")
    for spectrogram_file in tqdm(spectrogram_files):
        spectrogram_path = os.path.join(input_folder, spectrogram_file)
        
        # 加载梅尔频谱图并转换为 GPU tensor
        spectrogram = load_spectrogram(spectrogram_path)
        
        # 展平频谱图以作为聚类输入特征
        flattened_spectrogram = spectrogram.flatten().cpu().numpy()  # 确保转回 CPU 进行 KMeans 聚类
        features.append(flattened_spectrogram)
        file_names.append(spectrogram_file)

    # 将特征转换为 NumPy 数组
    features = np.array(features)

    # 使用 KMeans 进行聚类（在 CPU 上进行）
    print(f"Clustering into {NUM_CLUSTERS} clusters...")
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42).fit(features)

    # 获取聚类标签
    labels = kmeans.labels_

    # 将聚类结果保存到 JSON 文件
    print(f"Saving cluster mapping to {output_json} ...")
    cluster_mapping = {}
    for i, label in enumerate(labels):
        cluster_mapping[file_names[i]] = int(label)

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(cluster_mapping, f, ensure_ascii=False, indent=4)

    print("Clustering complete. Cluster mapping saved.")

# 主函数
if __name__ == "__main__":
    cluster_spectrograms()


