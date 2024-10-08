├── raw_audio/ # 原始音频文件
├── pure_audio/ # 清洗后音频文件
├── spectrogram_images/ # 梅尔频谱图可视化图像
├── spectrograms/ # 梅尔频谱图数值数据
├── data.json # 元数据和标注信息（待生成）
└── readme.txt # 本文件

数据处理说明
音频提取：使用FFmpeg从FLV视频中提取WAV音频。
音频聚类：使用librosa提取，使用 KMeans 进行聚类
频谱图生成：使用librosa库生成梅尔频谱图。
文本提取：使用OpenCV和PaddleORC进行OCR字幕提取。
情感标注：使用大型语言模型对话机器人基于提取的文本进行标注。
角色标签：通过对音频进行聚类分析获得。

频谱图数据（.npy文件）是模型训练的主要输入。
图像格式的频谱图（.png文件）主要用于可视化和检查。
原始WAV文件保留用于可能的未来分析和实验。

使用NumPy库加载.npy文件进行模型训练。
结合data.json中的文本和标签信息进行多模态分析。
可以使用原始音频进行额外的特征提取或分析。
