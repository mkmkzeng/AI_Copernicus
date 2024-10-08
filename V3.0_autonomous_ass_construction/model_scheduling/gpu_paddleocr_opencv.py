import cv2
import os
import time
from paddleocr import PaddleOCR
from tqdm import tqdm
import numpy as np
from difflib import SequenceMatcher  # 用于比较相似度
import re  # 用于处理文本中的特殊符号

# 初始化PaddleOCR，启用GPU
ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=True)

# 视频文件路径和ASS文件路径————提取完毕应当封存————————————————————————————————————————————————————————————————————
# video_files_set = '../../Video_file_set'
# ass_files_set = '../ass_file_set'
#——————————————————————————————————————————————————————————————————————————————————————————————————————————————

# 每秒采样帧数
FRAME_RATE = 10

# 字体大小阈值（作为高度的百分比）
FONT_SIZE_THRESHOLD = 0.045  # 筛掉背景字幕的阈值

# 置信度过滤阈值
CONFIDENCE_THRESHOLD = 0.98  # 置信度阈值

# 文本相似度阈值，用于合并处理（70%）
SIMILARITY_THRESHOLD = 0.70

# 最小文本长度阈值，短于3个字的文本将被过滤掉
MIN_TEXT_LENGTH = 3

# 部分重合合并触发的下限阈值（用于补集判断）
PARTIAL_OVERLAP_THRESHOLD = 0.40

# 子集匹配的重合度范围
SUBSET_MIN_THRESHOLD = 0.60
SUBSET_MAX_THRESHOLD = 0.80

def similar(a, b):
    """使用SequenceMatcher判断两个字符串的相似度，返回相似度比例"""
    return SequenceMatcher(None, a, b).ratio()

def normalize_text(text):
    """规范化OCR提取的字幕文本，去除所有符号、空白字符，保留纯文字"""
    # 移除所有非汉字字符，包括符号、空格等
    text = re.sub(r'[^\w]', '', text)
    return text.strip()

def is_subset(text1, text2):
    """判断是否文本的字符集合是另一个的子集（无关顺序）"""
    common_length = len(set(text1) & set(text2))
    return (SUBSET_MIN_THRESHOLD <= common_length / len(set(text1)) <= SUBSET_MAX_THRESHOLD or 
            SUBSET_MIN_THRESHOLD <= common_length / len(set(text2)) <= SUBSET_MAX_THRESHOLD)

def partial_overlap(a, b):
    """返回两个字符串的重合度和补集长度，字符顺序无关"""
    if a is None or b is None:  # 添加None检查
        return 0, max(len(a or ""), len(b or ""))  # 如果是None，返回0的重合度和长度差
    a_set, b_set = set(a), set(b)
    intersection_len = len(a_set & b_set)  # 交集长度
    union_len = len(a_set | b_set)  # 并集长度
    overlap_ratio = intersection_len / max(len(a_set), len(b_set))
    
    # 计算补集部分的长度（即非重合部分）
    non_overlap_len = union_len - intersection_len
    
    return overlap_ratio, non_overlap_len

def filter_by_font_size_and_confidence(ocr_result, frame_height):
    """过滤出符合条件的字幕结果，包括字体大小和置信度"""
    filtered_result = []
    confidence_scores = []  # 每次处理时重置置信度记录

    for line in ocr_result:
        # 检查 line 是否包含四边形和文本
        if isinstance(line, list) and len(line) > 0:
            for word_info in line:
                if len(word_info) > 0 and isinstance(word_info[0], list):  # 确保包含四边形
                    box = word_info[0]  # 获取四边形坐标
                    confidence = word_info[1][1]  # 获取置信度
                    text = word_info[1][0]  # 获取文本内容
                    
                    # 提取 y 坐标，确保 box 是有效的四边形
                    y_coords = [point[1] for point in box if isinstance(point, list) and len(point) == 2]
                    
                    if y_coords and len(y_coords) >= 2:  # 确保有有效的 y 坐标
                        text_height = max(y_coords) - min(y_coords)
                        # 检查字体大小和置信度，只有符合条件的才保留
                        if text_height > frame_height * FONT_SIZE_THRESHOLD and confidence >= CONFIDENCE_THRESHOLD:
                            filtered_result.append(word_info)
                            confidence_scores.append(confidence)  # 记录置信度
    return filtered_result, confidence_scores

def enhance_frame(frame):
    """增强视频帧的可读性以提高OCR识别的精度"""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 应用轻微的锐化处理
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_frame = cv2.filter2D(gray_frame, -1, kernel)
    return sharpened_frame

def extract_subtitles(video_path, ass_output_path):
    """从视频中提取字幕并生成ASS文件"""
    video_capture = cv2.VideoCapture(video_path)
    
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frame_interval = fps // FRAME_RATE
    
    subtitles = []
    confidence_scores = []
    frame_index = 0
    last_text = ""
    start_timestamp = None
    end_timestamp = None  # 记录当前字幕的结束时间
    
    bottom_crop_start = int(frame_height * 0.89)
    bottom_crop_end = frame_height
    left_crop = int(frame_width * 0.10)
    right_crop = int(frame_width * 0.90)

    pbar = tqdm(total=total_frames // frame_interval, desc="Processing frames", unit="frame")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            # 使用增强函数处理帧
            enhanced_frame = enhance_frame(frame)
            # 裁剪增强后的帧
            cropped_frame = enhanced_frame[bottom_crop_start:bottom_crop_end, left_crop:right_crop]

            bottom_result = ocr.ocr(cropped_frame)
            filtered_result, frame_confidences = filter_by_font_size_and_confidence(bottom_result, frame_height)

            confidence_scores.extend(frame_confidences)

            if filtered_result:
                try:
                    current_text = ''.join([normalize_text(word[1][0]) for word in filtered_result])
                except Exception as e:
                    print(f"Error processing OCR result: {e}")
                    current_text = ""

                if len(current_text) < MIN_TEXT_LENGTH:
                    current_text = ""

                if last_text == "" and current_text:
                    start_timestamp = video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000

                if current_text:
                    subtitles.append((start_timestamp, video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000, current_text))

                last_text = current_text
            else:
                if last_text:
                    end_timestamp = video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    subtitles.append((start_timestamp, end_timestamp, last_text))
                    last_text = ""

            pbar.update(1)

        frame_index += 1

    pbar.close()
    video_capture.release()
    generate_ass(subtitles, ass_output_path)
    
    # 计算并打印置信度信息
    if confidence_scores:
        min_conf = np.min(confidence_scores)
        max_conf = np.max(confidence_scores)
        avg_conf = np.mean(confidence_scores)
        print(f"最低置信度: {min_conf:.2f}, 最高置信度: {max_conf:.2f}, 平均置信度: {avg_conf:.2f}")
    else:
        print("没有置信度数据")

def generate_ass(subtitles, output_path):
    """生成ASS字幕文件，去重并合并相似的字幕，保留最长文本"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('[Script Info]\n')
        f.write('Title: Auto-generated Subtitles\n')
        f.write('ScriptType: v4.00+\n')
        f.write('Collisions: Normal\n')
        f.write('PlayDepth: 0\n')
        f.write('[V4+ Styles]\n')
        f.write('Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, '
                'Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n')
        f.write('Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,1,0,2,10,10,10,1\n')
        f.write('[Events]\n')
        f.write('Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n')

        previous_text = None
        previous_start_time = None
        previous_end_time = None

        for start_time, end_time, text in subtitles:
            # 检查时间间隔是否小于 00:00:00.45
            time_difference = start_time - previous_end_time if previous_end_time is not None else None

            # 检查相似度或子集匹配，或时间间隔小于 00:00:00.45
            if previous_text and (
                similar(previous_text, text) > SIMILARITY_THRESHOLD or 
                is_subset(previous_text, text) or 
                (time_difference is not None and time_difference < 0.45)
            ):
                previous_end_time = end_time
                # 保留最长的文本
                if len(text) > len(previous_text):
                    previous_text = text

            # 补集判断逻辑
            else:
                overlap_ratio, non_overlap_len = partial_overlap(previous_text, text)
                if PARTIAL_OVERLAP_THRESHOLD <= overlap_ratio <= SIMILARITY_THRESHOLD and non_overlap_len < 5:
                    previous_end_time = end_time
                    if len(text) > len(previous_text):
                        previous_text = text
                else:
                    # 如果有前一个字幕，先将其写入文件
                    if previous_text is not None:
                        start_time_str = f"{time.strftime('%H:%M:%S', time.gmtime(previous_start_time))}.{int((previous_start_time % 1) * 100):02d}"
                        end_time_str = f"{time.strftime('%H:%M:%S', time.gmtime(previous_end_time))}.{int((previous_end_time % 1) * 100):02d}"
                        f.write(f'Dialogue: 0,{start_time_str},{end_time_str},Default,,0,0,0,,{previous_text}\n')

                    # 更新为当前字幕
                    previous_text = text
                    previous_start_time = start_time
                    previous_end_time = end_time

        # 写入最后一条字幕
        if previous_text is not None:
            start_time_str = f"{time.strftime('%H:%M:%S', time.gmtime(previous_start_time))}.{int((previous_start_time % 1) * 100):02d}"
            end_time_str = f"{time.strftime('%H:%M:%S', time.gmtime(previous_end_time))}.{int((previous_end_time % 1) * 100):02d}"
            f.write(f'Dialogue: 0,{start_time_str},{end_time_str},Default,,0,0,0,,{previous_text}\n')




def process_videos():
    """处理视频目录中的所有视频文件"""
    
    # 通过正则表达式从文件名中提取数字，并使用自然顺序进行排序
    def natural_sort_key(filename):
        return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', filename)]
    
    # 对目录中的文件进行自然排序
    flv_files = sorted([f for f in os.listdir(video_files_set) if f.endswith('.flv')], key=natural_sort_key)

    # 遍历排序后的文件
    for filename in flv_files:
        video_path = os.path.join(video_files_set, filename)
        ass_output_path = os.path.join(ass_files_set, f'{os.path.splitext(filename)[0]}.ass')
        print(f"Processing video: {filename}")
        extract_subtitles(video_path, ass_output_path)

if __name__ == '__main__':
    process_videos()





