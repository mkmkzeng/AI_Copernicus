import os
import json
import subprocess
import sys
import re

# 文件路径
folder_path = "./"  # 当前工作目录
last_path = "../"  # 上一个目录
dataset_folder = os.path.join(folder_path, "dataset")
audio_folder = os.path.join(dataset_folder, "pure_audio")
video_folder = os.path.join(last_path, "Video_file_set")  # 存放 .flv 文件的文件夹
title_folder = os.path.join(folder_path, "ass_file_set")  # 存放 .ass 文件的文件夹
json_output = os.path.join(dataset_folder, "audio.json")

# 修改默认编码为 UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# 时间格式转换函数 "0:00:19.29" -> 秒
def convert_time_to_seconds(time_str):
    h, m, s = time_str.split(':')
    s, ms = s.split('.')
    return int(h) * 3600 + int(m) * 60 + int(s) + float('0.' + ms)

# 获取视频时长
def get_video_duration(flv_file):
    command = ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
               "-of", "default=noprint_wrappers=1:nokey=1", flv_file]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return float(result.stdout) if result.returncode == 0 else 0

# 累加时间戳处理
def adjust_timestamps(subtitles, base_time):
    adjusted_subtitles = []
    for (start_time, end_time, text) in subtitles:
        start_seconds = convert_time_to_seconds(start_time) + base_time
        end_seconds = convert_time_to_seconds(end_time) + base_time
        adjusted_subtitles.append((start_seconds, end_seconds, text))
    return adjusted_subtitles, end_seconds

# 映射时间戳到 [0, 1] 区间
def map_timestamps(adjusted_subtitles, total_duration):
    mapped_subtitles = []
    for (start_time, end_time, text) in adjusted_subtitles:
        mapped_start = start_time / total_duration
        mapped_end = end_time / total_duration
        mapped_subtitles.append((mapped_start, mapped_end, text))
    return mapped_subtitles

def parse_ass_file(ass_file_path):
    subtitles = []
    with open(ass_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith("Dialogue:"):
                parts = line.split(',', 9)
                if len(parts) >= 10:
                    start_time = parts[1].strip()
                    end_time = parts[2].strip()
                    text = parts[9].strip()
                    if text:
                        subtitles.append((start_time, end_time, text))
    return subtitles

# 自然排序的 key 函数（按数字排序）
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

# 清除旧的 JSON 文件
def clear_json_file(json_output):
    if os.path.exists(json_output):
        try:
            os.remove(json_output)
            print(f"已删除旧的 JSON 文件: {json_output}")
        except Exception as e:
            print(f"删除 JSON 文件失败: {e}")
    else:
        print("没有找到旧的 JSON 文件。")

# 生成统一的 JSON 文件
def generate_json():
    # 清除旧的 JSON 文件
    clear_json_file(json_output)
    
    data = []
    total_duration = 0  # 累加视频总时长
    base_time = 0  # 累加时间戳
    
    # 获取所有 .flv 文件和 .ass 文件，并按自然顺序排序
    video_files = sorted([f for f in os.listdir(video_folder) if f.endswith('.flv')], key=natural_sort_key)
    title_files = sorted([f for f in os.listdir(title_folder) if f.endswith('.ass')], key=natural_sort_key)

    # 创建文件名前缀的映射，确保文件名正确匹配
    video_prefix_map = {os.path.splitext(f)[0]: f for f in video_files}
    title_prefix_map = {os.path.splitext(f)[0]: f for f in title_files}

    # 计算所有视频的总时长
    for flv_file in video_files:
        flv_file_path = os.path.join(video_folder, flv_file)
        total_duration += get_video_duration(flv_file_path)

    for prefix in video_prefix_map.keys():
        if prefix in title_prefix_map:
            flv_file = video_prefix_map[prefix]
            ass_file = title_prefix_map[prefix]

            flv_file_path = os.path.join(video_folder, flv_file)
            ass_file_path = os.path.join(title_folder, ass_file)

            # 获取当前 .flv 文件的视频时长
            video_duration = get_video_duration(flv_file_path)

            # 提取 .ass 文件中的时间戳和文本
            subtitles = parse_ass_file(ass_file_path)
            adjusted_subtitles, last_end_time = adjust_timestamps(subtitles, base_time)

            # 更新 base_time 为当前 .ass 文件结束的时间
            base_time += video_duration

            # 映射时间戳到 [0, 1]
            mapped_subtitles = map_timestamps(adjusted_subtitles, total_duration)

            # 对每条字幕生成对应的 JSON 项
            for j, (start_time, end_time, text_original) in enumerate(mapped_subtitles):
                audio_filename = os.path.join(audio_folder, f"{prefix}_{j+1:03d}.wav")

                if os.path.exists(audio_filename):
                    data.append({
                        "audio_file": audio_filename,
                        "text_original": text_original,
                        "text_processed": "",  # 留空，等待模型生成的文本
                        "start_time": start_time,
                        "end_time": end_time,
                        "character": "",  # 留空，等待聚类模型写入角色信息
                        "emotion_category": ""  # 留空，等待标注情感类别信息
                    })
                else:
                    # 保留信息，即使没有对应的 .wav 文件
                    data.append({
                        "audio_file": "",
                        "text_original": text_original,
                        "text_processed": "",
                        "start_time": start_time,
                        "end_time": end_time,
                        "character": "",
                        "emotion_category": ""
                    })
                    print(f"警告: {audio_filename} 文件未找到，生成对应的 JSON 项但 audio_file 为空。")

        else:
            print(f"警告: 未找到与 {prefix} 匹配的 .ass 文件，跳过该文件。")

    # 保存生成的 JSON 文件
    with open(json_output, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    generate_json()



