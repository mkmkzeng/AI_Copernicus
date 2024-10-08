import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from tqdm import tqdm

# 检查是否可以使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 输入和输出文件夹
input_folder = './dataset/pure_audio'
npy_output_folder = './dataset/spectrograms'
png_output_folder = './dataset/spectrogram_images'

# 创建输出文件夹（如果不存在）
os.makedirs(npy_output_folder, exist_ok=True)
os.makedirs(png_output_folder, exist_ok=True)

# 音频处理参数
SR = 22050  # 采样率
N_MELS = 128  # 梅尔频率带数量
HOP_LENGTH = 512  # 帧移长度
WIN_LENGTH = 2048  # 窗口长度
BATCH_SIZE = 32  # 批处理的大小，视显存而定

# 定义MelSpectrogram的转换器
mel_spectrogram_transform = MelSpectrogram(
    sample_rate=SR,
    n_mels=N_MELS,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH
).to(device)

# 定义将功率谱转换为分贝
amplitude_to_db = AmplitudeToDB().to(device)

# 批量加载音频
def batch_load_audio(wav_files, input_folder):
    """批量加载音频文件"""
    waveforms = []
    sample_rates = []
    for wav_file in wav_files:
        wav_path = os.path.join(input_folder, wav_file)
        waveform, sr = torchaudio.load(wav_path)  # 加载音频文件
        waveforms.append(waveform)
        sample_rates.append(sr)
    
    # 将所有音频文件拼接为一个大批次
    waveforms = torch.cat(waveforms, dim=0)
    
    return waveforms, sample_rates

# 使用 torchaudio 生成梅尔频谱图的函数
def generate_mel_spectrograms(waveforms, sample_rates):
    """生成梅尔频谱图并使用GPU加速"""
    mel_spectrograms = []
    for i, waveform in enumerate(waveforms):
        # 如果采样率不一致，重采样到指定的 SR
        if sample_rates[i] != SR:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rates[i], new_freq=SR).to(device)
            waveform = resampler(waveform)

        # 将波形移动到GPU并生成梅尔频谱图
        waveform = waveform.to(device)
        mel_spectrogram = mel_spectrogram_transform(waveform)
        mel_spectrogram_db = amplitude_to_db(mel_spectrogram)  # 转换为dB尺度
        mel_spectrograms.append(mel_spectrogram_db.cpu().numpy())
    
    return mel_spectrograms

# 保存频谱图数据和可视化图像
def save_spectrogram_data_and_images(wav_files, mel_spectrograms):
    """保存梅尔频谱图数据和图像"""
    for i, wav_file in enumerate(wav_files):
        npy_output_path = os.path.join(npy_output_folder, wav_file.replace('.wav', '.npy'))
        png_output_path = os.path.join(png_output_folder, wav_file.replace('.wav', '.png'))
        
        # 检查文件是否已经存在
        if os.path.exists(npy_output_path) and os.path.exists(png_output_path):
            print(f"Skipping {wav_file}, files already exist.")
            continue
        
        mel_spectrogram = mel_spectrograms[i]

        # 保存频谱图数据到 .npy 文件
        try:
            np.save(npy_output_path, mel_spectrogram)
        except Exception as e:
            print(f"Error saving {npy_output_path}: {e}")
        
        # 生成频谱图图像并保存为 .png 文件
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_spectrogram[0], aspect='auto', origin='lower')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-Spectrogram')
        plt.tight_layout()
        plt.savefig(png_output_path)
        plt.close()

# 批处理所有 WAV 文件
def batch_process_audio_files(audio_files, batch_size):
    """批量处理WAV文件"""
    num_batches = len(audio_files) // batch_size + int(len(audio_files) % batch_size != 0)

    for batch_idx in range(num_batches):
        # 获取当前批次的文件列表
        batch_files = audio_files[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        print(f"Processing batch {batch_idx + 1}/{num_batches}, files: {batch_files}")
        
        # 批量加载音频文件
        waveforms, sample_rates = batch_load_audio(batch_files, input_folder)
        
        # 生成批量梅尔频谱图
        mel_spectrograms = generate_mel_spectrograms(waveforms, sample_rates)
        
        # 保存频谱图数据和图像
        save_spectrogram_data_and_images(batch_files, mel_spectrograms)

# 处理所有 WAV 文件
audio_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]

print(f"Processing {len(audio_files)} audio files in batches of {BATCH_SIZE}...")

# 批量处理音频文件
batch_process_audio_files(audio_files, BATCH_SIZE)

print("Processing complete. Spectrograms saved in .npy and .png format.")

