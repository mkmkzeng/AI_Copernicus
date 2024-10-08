import os
import json
import pysubs2
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载RoBERTa模型和分词器
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=4)  # 4分类情感任务
model.to(device)

# 自定义情感分类函数
def classify_emotion(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
    emotion_map = {0: "Happy", 1: "Sad", 2: "Angry", 3: "Neutral"}
    return [emotion_map[pred] for pred in predictions]

# 全局情感标注
def get_global_emotion(texts):
    return classify_emotion([" ".join(texts)])[0]  # 对三条文本拼接后的全局情感标注

# 局部情感标注
def get_local_emotion(text):
    return classify_emotion([text])[0]

# 将ASS格式时间转换为秒数，保留毫秒部分
def convert_time_format(ass_time):
    """将ASS格式的时间戳转换为秒数"""
    h, m, s = ass_time.split(":")
    s, ms = s.split(".")  # 处理毫秒部分
    return int(h) * 3600 + int(m) * 60 + int(s) + float(f"0.{ms}")

# 处理ASS文件
def process_ass_files(ass_files):
    result_json = []
    
    # 按文件名顺序读取
    for ass_file in sorted(ass_files, key=lambda x: int(x.split('.')[0])):
        subs = pysubs2.load(ass_file, encoding="utf-8")
        dialogues = [d for d in subs if d.type == "Dialogue"]
        
        # 每3条台词作为一个窗口进行处理
        for i in range(0, len(dialogues), 3):
            group = dialogues[i:i+3]
            
            # 获取三条台词文本，若不足三条则补全为空字符串
            texts = [dialogue.text.strip() for dialogue in group]
            while len(texts) < 3:
                texts.append("")  # 确保窗口内有三条文本
            
            # 获取全局情感标注
            global_emotion = get_global_emotion(texts)
            
            for dialogue in group:
                start_time = convert_time_format(dialogue.start.to_string())
                end_time = convert_time_format(dialogue.end.to_string())
                text_original = dialogue.text.strip()

                # 获取局部情感标注
                emotion_category = get_local_emotion(text_original)
                
                # 构造JSON结构
                result_json.append({
                    "text_original": text_original,
                    "start_time": start_time,
                    "end_time": end_time,
                    "emotion_category": emotion_category,
                    "global_emotion": global_emotion
                })
    
    return result_json

# 保存JSON文件
def save_json_file(output_path, data):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 主函数
def main():
    # 假设ASS文件存储在ass_file_set文件夹中
    ass_folder = "../ass_file_set"
    ass_files = [os.path.join(ass_folder, f) for f in os.listdir(ass_folder) if f.endswith('.ass')]
    
    # 处理ASS文件并生成JSON数据
    result_data = process_ass_files(ass_files)
    
    # 保存生成的JSON文件
    output_path = "output.json"  # 输出JSON文件路径
    save_json_file(output_path, result_data)

if __name__ == "__main__":
    main()
