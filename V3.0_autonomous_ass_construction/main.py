# 1.‘gpu_pytesseract_opencv.py’调用大模型对‘video_file_set’中的flv进行orc光学检测然后生成ass文件存入‘ass_file_set’。
# 2.‘extract_wav.py’根据‘ass_file_set’中的ass与‘video_file_set’中的flv进行wav文件提取并保存进‘dataset’里的‘raw_audio’。
# 3.‘construct_audio_json.py’根据‘ass_file_set’中的ass与‘video_file_set’中的flv与‘dataset’里的‘raw_audio’生成‘data.json’保存在‘dataset’中。
# 4.‘audio_filter.py’对‘raw_audio’中的wav文件进行清洗然后存入‘pure_audio’