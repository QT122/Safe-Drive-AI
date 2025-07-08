from pathlib import Path
from torch.utils.data import Dataset
import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

def predict_single_sequence(model, dataset, idx, device, class_names=None, visualize=True):
    model.eval()

    with torch.no_grad():
        window, label, name = dataset[idx]  # 取出数据
        input_tensor = window.unsqueeze(0).to(device)  # [1, T, D]
        output = model(input_tensor)  # [1, num_classes]
        probs = F.softmax(output, dim=1).cpu().numpy()[0]  # 转为概率分布
        pred = int(probs.argmax())

    # 打印预测结果
    print(f" 序列名: {name}")
    print(f" 真实标签: {label.item()} -> {class_names[label.item()] if class_names else label.item()}")
    print(f" 预测标签: {pred} -> {class_names[pred] if class_names else pred}")
    print(f" 概率分布: {probs.round(4)}")

    # 可视化分类概率
    if visualize and class_names:
        plt.figure(figsize=(6, 4))
        plt.bar(class_names, probs, color='skyblue')
        plt.ylabel("概率")
        plt.title(f"序列 {name} 的分类预测分布")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=12, hidden_size=32, output_size=5):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        final_step = lstm_out[:, -1, :]
        logits = self.fc(final_step)
        return logits

'''
#class 名称
class_names = ['no_belt', 'disturb', 'phone', 'eat/drink', 'normal']
# 加载训练好的模型
model = LSTMClassifier(input_size=12, hidden_size=32, output_size=5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("/Users/zhou/Desktop/Driving/python/models/best_lstm.pth"))
model.to(device)


这里有点伪代码，可能对你有帮助：

buffer = []  # 存储每帧的12维特征
# 假设在你的主循环中不断采集：
buffer.append(current_12d_feature)
if len(buffer) >= 100:
    sequence = buffer[-100:]  # 保留最近的100帧
    input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)  # shape: [1, 100, 12]
    model.eval()  # 设置模型为推理模式
    with torch.no_grad():
        input_tensor = input_tensor.to(device)  # 放到GPU或CPU
        output = model(input_tensor)  # shape: [1, 5]，预测每个类别的概率/得分
        predicted_label = torch.argmax(output, dim=1).item()

'''
