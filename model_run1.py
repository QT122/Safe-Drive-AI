import os
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import mediapipe as mp
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ======= 模型加载 =======
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                            refine_landmarks=True, min_detection_confidence=0.5)


def analyze_fatigue(image_path, features_path, ear_mar_path, show_image=True):

    # ======= 读取图像 =======
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"无法读取图像：{image_path}")
        return {"error": "图像无法读取"}
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = image_rgb.shape

    # ======= 人脸识别函数 =======
    def recognize_face(image_rgb):
        features = np.load(features_path, allow_pickle=True).item()
        pil_img = Image.fromarray(image_rgb)
        face = mtcnn(pil_img)
        if face is None:
            return "Unknown", None
        with torch.no_grad():
            emb = facenet(face.unsqueeze(0).to(device)).cpu().numpy()
        best_score = -1
        best_name = "Unknown"
        for name, db_feat in features.items():
            sim = cosine_similarity(emb, db_feat.reshape(1, -1))[0][0]
            if sim > best_score:
                best_score = sim
                best_name = name
        return (best_name, best_score) if best_score >= 0.6 else ("Unknown", None)

    # ======= EAR / MAR计算函数 =======
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    MOUTH = [[61, 291], [39, 181], [0, 17], [269, 405]]

    def get_ear(landmarks, eye_idx):
        coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_idx]
        A = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
        B = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
        C = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))
        return (A + B) / (2.0 * C)

    def get_mar(landmarks):
        coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for pair in MOUTH for i in pair]
        N1 = np.linalg.norm(np.array(coords[2]) - np.array(coords[3]))
        N2 = np.linalg.norm(np.array(coords[4]) - np.array(coords[5]))
        N3 = np.linalg.norm(np.array(coords[6]) - np.array(coords[7]))
        D = np.linalg.norm(np.array(coords[0]) - np.array(coords[1]))
        return (N1 + N2 + N3) / (3 * D)

    # ======= 执行检测 =======
    name, sim = recognize_face(image_rgb)
    if name == "Unknown":
        return {"error": "无法识别身份"}

    ear_mar = pd.read_csv(ear_mar_path, index_col="person_id")
    if name not in ear_mar.index:
        return {"error": "未找到对应阈值"}

    EAR_THRESH = ear_mar.loc[name, "avg_EAR"]
    MAR_THRESH = ear_mar.loc[name, "avg_MAR"]

    results = face_mesh.process(image_rgb)
    if not results.multi_face_landmarks:
        return {"error": "未检测到人脸网格"}

    landmarks = results.multi_face_landmarks[0].landmark
    ear_l = get_ear(landmarks, LEFT_EYE)
    ear_r = get_ear(landmarks, RIGHT_EYE)
    mar = get_mar(landmarks)
    avg_ear = (ear_l + ear_r) / 2.0

    # 疲劳状态判断
    reasons = []
    if avg_ear < EAR_THRESH:
        reasons.append("闭眼")
    if mar > MAR_THRESH:
        reasons.append("打哈欠")

    result_dict = {
        "name": name,
        "similarity": sim,
        "avg_ear": avg_ear,
        "mar": mar,
        "ear_thresh": EAR_THRESH,
        "mar_thresh": MAR_THRESH,
        "is_drowsy": bool(reasons),
        "reasons": reasons
    }

######可视化
    if show_image:
        cv2.putText(image_bgr, f"User: {name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(image_bgr, f"EAR: {avg_ear:.2f}  MAR: {mar:.2f}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 0), 2)
        for i in LEFT_EYE + RIGHT_EYE + [i for pair in MOUTH for i in pair]:
            cx, cy = int(landmarks[i].x * w), int(landmarks[i].y * h)
            cv2.circle(image_bgr, (cx, cy), 2, (0, 255, 0), -1)
        cv2.imshow("Fatigue Detection Result", image_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result_dict

def run_models(img_path, features_path, ear_mar_path):
    from PIL import Image
    img = Image.open(img_path)
    width, height = img.size  # width=640, height=480
    # 计算宽度方向的裁剪边界（居中裁剪）
    left = (width - height) // 2
    right = left + height  # left+480
    # 裁剪为中间的 480x480
    img_cropped = img.crop((left, 0, right, height))
    # 缩放到 640x640
    img = img_cropped.resize((640, 640))

    # === 模型 1 ===
    results1 = model1(img, verbose=False)
    classes1 = results1[0].boxes.cls.cpu().tolist()
    confs1 = results1[0].boxes.conf.cpu().tolist()

    score1_0, score1_1, score1_2 = 0.0, 0.0, 0.0
    for cls, conf in zip(classes1, confs1):
        if int(cls) == 0:
            score1_0 = max(score1_0, conf)
        elif int(cls) == 1:
            score1_1 = max(score1_1, conf)
        elif int(cls) == 2:
            score1_2 = max(score1_2, conf)

    # === 模型 2 ===
    results2 = model2(img, verbose=False)
    classes2 = results2[0].boxes.cls.cpu().tolist()
    confs2 = results2[0].boxes.conf.cpu().tolist()

    score2_0, score2_1, score2_2, score2_3 = 0.0, 0.0, 0.0, 0.0
    for cls, conf in zip(classes2, confs2):
        idx = int(cls)
        if idx == 0:
            score2_0 = max(score2_0, conf)
        elif idx == 1:
            score2_1 = max(score2_1, conf)
        elif idx == 2:
            score2_2 = max(score2_2, conf)
        elif idx == 3:
            score2_3 = max(score2_3, conf)

    # === 模型 3（疲劳检测）===
    try:
        result = analyze_fatigue(
            image_path=str(img_path),
            features_path=features_path,
            ear_mar_path=ear_mar_path,
            show_image=False
        )
        score3_0 = result.get("avg_ear", 0)
        score3_1 = result.get("ear_thresh", 0)
        score3_2 = result.get("mar", 0)
        score3_3 = result.get("mar_thresh", 0)
        score3_4 = int(result.get("is_drowsy", 0))

    except Exception as e:
        print(f"[Error] analyze_fatigue failed on {img_path}: {e}")
        score3_0 = score3_1 = score3_2 = score3_3 = 0
        score3_4 = 0

    final_result = [
        score1_0, score1_1, score1_2,
        score2_0, score2_1, score2_2, score2_3,
        score3_0, score3_2, score3_4
    ]
    return {
        "features": [round(float(x), 4) for x in final_result],
        "avg_ear": score3_0,
        "ear_thresh": score3_1,
        "mar": score3_2,
        "mar_thresh": score3_3,
        "is_drowsy": score3_4,
        "name": result.get("name", "Unknown")
    }


# 模型加载
model1 = YOLO(r"D:\python\演示用\.venv\后端\object.pt") #这里放物体识别路径
model2 = YOLO(r'D:\python\演示用\.venv\后端\body.pt')  # 4类，这里放姿势识别路径



