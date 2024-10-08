import os
import subprocess
import sys

# 修改默认编码为 UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# 文件路径
folder_path = "./"  # 当前工作目录
last_path = '../'   #上一工作目录
dataset_folder = os.path.join(folder_path, "dataset")
audio_folder = os.path.join(dataset_folder, "raw_audio")
video_folder = os.path.join(last_path, "Video_file_set")  # 存放 .flv 文件的文件夹
title_folder = os.path.join(folder_path, "ass_file_set")  # 存放 .ass 文件的文件夹

# 创建必要的文件夹
os.makedirs(audio_folder, exist_ok=True)

# 清理音频文件夹中所有的 WAV 文件
def clear_audio_folder(audio_folder):
    for filename in os.listdir(audio_folder):
        file_path = os.path.join(audio_folder, filename)
        try:
            if filename.endswith(".wav"):
                os.remove(file_path)
                print(f"已删除: {file_path}")
        except Exception as e:
            print(f"删除文件失败: {file_path}, 错误: {e}")

def convert_time_to_seconds(time_str):
    # 移除末尾的 .00
    if ".00" in time_str:
        time_str = time_str.replace(".00", "")
    
    h, m, s = time_str.split(':')

    # 检查秒数是否有小数部分
    if '.' in s:
        s, ms = s.split('.')
        return int(h) * 3600 + int(m) * 60 + int(s) + float('0.' + ms)
    else:
        return int(h) * 3600 + int(m) * 60 + int(s)

# 解析 .ass 文件提取时间信息
def parse_ass_file(ass_file_path):
    subtitles = []
    with open(ass_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith("Dialogue:"):
                parts = line.split(',')
                if len(parts) >= 3:
                    # 移除多余的 .00 部分
                    start_time = parts[1].replace('.00', '')
                    end_time = parts[2].replace('.00', '')
                    subtitles.append((start_time, end_time))
    return subtitles

# 使用 ffmpeg 提取音频片段
def extract_audio_segment(flv_file, start_time, end_time, output_file):
    start_seconds = convert_time_to_seconds(start_time)
    end_seconds = convert_time_to_seconds(end_time)
    duration = end_seconds - start_seconds

    # ffmpeg 提取命令
    command = [
        "ffmpeg", "-y", "-i", flv_file, "-ss", str(start_seconds), "-t", str(duration),
        "-q:a", "0", "-map", "a", output_file
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# 处理所有视频和对应的ass文件
def process_videos():
    # 遍历 Video_file_set 和 Title_file_set 文件夹中的文件
    video_files = sorted(os.listdir(video_folder))  # 获取所有 .flv 文件
    title_files = sorted(os.listdir(title_folder))  # 获取所有 .ass 文件

    for flv_file, ass_file in zip(video_files, title_files):
        flv_file_path = os.path.join(video_folder, flv_file)
        ass_file_path = os.path.join(title_folder, ass_file)
        
        # 确保 .flv 和 .ass 文件匹配
        if flv_file.endswith('.flv') and ass_file.endswith('.ass'):
            subtitles = parse_ass_file(ass_file_path)
            for j, (start_time, end_time) in enumerate(subtitles):
                audio_filename = os.path.join(audio_folder, f"{flv_file[:-4]}_{j+1:03d}.wav")
                extract_audio_segment(flv_file_path, start_time, end_time, audio_filename)
                print(f"提取了音频: {audio_filename}")

if __name__ == "__main__":
    # 开始前清理旧的 .wav 文件
    clear_audio_folder(audio_folder)
    
    # 处理视频和字幕文件
    process_videos()