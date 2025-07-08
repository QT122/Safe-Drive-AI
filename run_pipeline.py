import os
import json
from glob import glob
from pathlib import Path
from model_run import run_models, analyze_fatigue
from LSTM_pre import LSTMClassifier
import torch.nn.functional as F
import os
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import mediapipe as mp
import os
import time
import cv2
from collections import deque
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import tempfile
import shutil
from collections import deque
from datetime import datetime
#  路径配置记得改
UPLOAD_DIR = r"D:\python\演示用\.venv\后端\uploads"
RESULT_PATH = r"D:\python\演示用\.venv\后端\result.json"
FEATURES_PATH = r"D:\python\演示用\.venv\后端\features.npy"
EAR_MAR_PATH = r"D:\python\演示用\.venv\后端\avg_ear_mar.csv"
LSTM_MODEL_PATH = r"D:\python\演示用\.venv\后端\best_lstm_v3.pth"
VIOLATION_LOG_PATH = r"D:\python\演示用\.venv\后端\violation_log.txt"
all_features = deque(maxlen=100)
first_write = True
# 用于记录5秒内的眨眼和打哈欠次数
# 用于记录过去5秒内的疲劳行为发生时间（单位：秒时间戳）
blink_window = deque()
yawn_window = deque()

def record_violation(msg, violations_list=None, image_name="unknown.jpg"):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted = f"[{timestamp}] Action detection：{msg}"
    print(formatted)
    with open(VIOLATION_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")
    if violations_list is not None:
        violations_list.append(("Action Detection", msg))

# 监控处理器
class SingleImageWatcher(FileSystemEventHandler):
    def __init__(self):
        self.last_processed_file = None
        self.new_result_count = 0  # 计数器

    def try_process(self, path):
        for _ in range(10):  # 最多尝试10次读取
            try:
                start=time.time()
                violations = []
                result_data = run_models(path, FEATURES_PATH, EAR_MAR_PATH)
                feats = result_data["features"]
                avg_ear = result_data["avg_ear"]
                avg_mar = result_data["mar"]
                # ============ 模型3（疲劳检测） ============
                try:
                    fatigue_result = analyze_fatigue(path, FEATURES_PATH, EAR_MAR_PATH, show_image=False)
                    is_drowsy = fatigue_result.get("is_drowsy", False)
                    reasons = fatigue_result.get("reasons", [])

                    # 判断是否闭眼或打哈欠
                    # === 记录最近5秒内的疲劳行为 ===
                    now = time.time()

                    # 保留最近5秒内的事件
                    # 保留最近5秒内的事件（原地修改）
                    blink_window_data = [t for t in blink_window if now - t <= 5]
                    yawn_window_data = [t for t in yawn_window if now - t <= 5]
                    blink_window.clear()
                    blink_window.extend(blink_window_data)
                    yawn_window.clear()
                    yawn_window.extend(yawn_window_data)

                    # 统计5秒内疲劳行为次数
                    blink_count = len(blink_window)
                    yawn_count = len(yawn_window)

                except Exception as e:
                    print("⚠️ 疲劳分析失败", e)
                    blink_count = yawn_count = 0

                all_features.append(feats)
                print(f"✅ 成功提取: {os.path.basename(path)}（当前检测到闭眼队列长度: {len(all_features)}）")
                print(feats)
                print(f"{time.time()-start}s")

                self.last_processed_file = path
                self.new_result_count+=1
                if self.new_result_count >= 5 and len(all_features) >= 5:


                    sequence_tensor = torch.tensor(all_features, dtype=torch.float32).unsqueeze(0).to(
                        device)  # shape [1, 60, 12]
                    with torch.no_grad():
                        start2=time.time()
                        output = lstm_model(sequence_tensor)
                        probs = F.softmax(output, dim=1).cpu().numpy()[0]
                        predicted_class = int(np.argmax(probs))
                        print("lstm result is:",predicted_class,time.time()-start2)
                    self.new_result_count = 0  # 重置计数器
                    # 其他违规行为判断
                    image_name = os.path.basename(path)

                    # 疲劳行为
                    if "闭眼" in reasons:
                        blink_window.append(now)
                        record_violation("Detected closing of eyes", violations, image_name)

                    if "打哈欠" in reasons:
                        yawn_window.append(now)
                        record_violation("Detected yawning", violations, image_name)

                    # 行为检测
                    if feats[0] > 0.2:
                        record_violation("Detected an object", violations, image_name)

                    if feats[1] < 0.5 and feats[2] > 0.5:
                        record_violation("Detected that the seat belt was not fastened.", violations, image_name)

                    if feats[3] > 0.4:
                        record_violation("Detected an extra hand", violations, image_name)

                    if feats[4] > 0.5 and feats[5] < 0.5:
                        record_violation("Detected that the hand has left the steering wheel", violations, image_name)

                    # 构造 JSON 输出结果
                    CLASS_NAMES = ["no_belt", "disturb", "violating object", "violating object", "normal"]
                    result = {
                        "predicted_class": predicted_class,
                        "class_name": CLASS_NAMES[predicted_class],
                        "class_probabilities": [round(float(p), 4) for p in probs],
                        "objects":int(feats[0]>0.3),
                        "with_seatbelt":int(feats[1]>0.3),
                        "without_seatbelt":int(feats[2]>0.5),
                        "extra_hand":int(feats[3]>0.1),
                        "hands_off":int(feats[4]>0.03),
                        "hands_on":int(feats[5]>0.1),
                        "wheel":feats[6],
                        "name": result_data.get("name", "Unknown"),
                        "avg_ear":avg_ear,
                        "avg_mar":avg_mar,
                        "blink_count": blink_count,
                        "yawn_count": yawn_count,
                        "violations": violations

                    }
                    with tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8',
                                                     dir=os.path.dirname(RESULT_PATH)) as tmpf:
                        json.dump(result, tmpf, ensure_ascii=False, indent=2)
                        temp_name = tmpf.name
                    shutil.move(temp_name, RESULT_PATH)
                    # 将所有违规信息写入日志
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    with open(VIOLATION_LOG_PATH, "a", encoding="utf-8") as f:
                        for category, message in violations:
                            f.write(f"[{timestamp}] {category}：{message}\n")

                return
            except Exception as e:
                time.sleep(0.2)
        print(f"❌ 图像读取失败: {os.path.basename(path)}")

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".jpg"):
            self.try_process(event.src_path)

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(".jpg"):
            if event.src_path != self.last_processed_file:
                self.try_process(event.src_path)

    def on_deleted(self, event):
        if not event.is_directory and event.src_path == self.last_processed_file:
            print(f"🗑️ 文件被删除: {os.path.basename(event.src_path)}")
            self.last_processed_file = None

# 启动监控
def start_watching(folder):
    observer = Observer()
    handler = SingleImageWatcher()
    observer.schedule(handler, folder, recursive=False)
    observer.start()
    print(f"👁️ 实时监控文件夹: {folder}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    # 分类标签
    CLASS_NAMES = ["no_belt", "disturb", "phone", "eat/drink", "normal"]
    # 初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)

    #  加载 LSTM
    lstm_model = LSTMClassifier(input_size=10, hidden_size=32, output_size=5).to(device)
    lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device))
    lstm_model.eval()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # ======= 模型加载 =======
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                                refine_landmarks=True, min_detection_confidence=0.5)
    start_watching(UPLOAD_DIR)

'''
# 加载100张图像
image_paths = sorted(glob(os.path.join(UPLOAD_DIR, "*.jpg")))[:60]
if len(image_paths) < 60:
    raise ValueError(" 上传图像不足60张，当前仅有 {} 张".format(len(image_paths)))

#  提取100张图的12维特征
all_features = []
for img_path in image_paths:
    feats = run_models(img_path, FEATURES_PATH, EAR_MAR_PATH)  # 返回12维
    all_features.append(feats)

print(len(all_features))

#  LSTM 分类预测
sequence_tensor = torch.tensor(all_features, dtype=torch.float32).unsqueeze(0).to(device)  # shape [1, 60, 12]
with torch.no_grad():
    output = lstm_model(sequence_tensor)
    probs = F.softmax(output, dim=1).cpu().numpy()[0]
    predicted_class = int(np.argmax(probs))

print(all_features[len(all_features)-1])
'''
'''
# 构造 JSON 输出结果
result = {
    "predicted_class": predicted_class,
    "class_name": CLASS_NAMES[predicted_class],
    "class_probabilities": [round(float(p), 4) for p in probs],
    "name": fatigue_result.get("name", "Unknown"),
    "similarity": round(float(fatigue_result.get("similarity", 0)), 4),
    "avg_ear": round(float(fatigue_result.get("avg_ear", 0)), 4),
    "mar": round(float(fatigue_result.get("mar", 0)), 4),
    "ear_thresh": round(float(fatigue_result.get("ear_thresh", 0)), 4),
    "mar_thresh": round(float(fatigue_result.get("mar_thresh", 0)), 4),
    "is_drowsy": fatigue_result.get("is_drowsy", False),
    "reasons": fatigue_result.get("reasons", [])
}

with open(RESULT_PATH, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

# 输出结果
print("\n[推理完成：统一处理60张图像]")
print(f" 识别类别：{CLASS_NAMES[predicted_class]}")
print(f" 疲劳状态：{'疲劳' if result['is_drowsy'] else '正常'}（理由：{result['reasons']}）")
print(f" 用户身份：{result['name']}（相似度：{result['similarity']}）")
print(f" JSON 已保存至：{RESULT_PATH}")
'''