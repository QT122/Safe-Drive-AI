import os
import time
import shutil

# 配置路径
A_DIR = r"C:\Users\吴\Desktop\code\python\safe_drive\safe_drive.v3i.yolov8_姿势组\test\images"  # 源图像文件夹
B_DIR = r"C:\Users\吴\Desktop\code\python\safe_drive\前端\uploads"  # 目标文件夹
TARGET_FILENAME = "latest.jpg"   # B文件夹中始终保存为该文件名


def get_image_list(folder):
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

def loop_copy_images():
    images = get_image_list(A_DIR)
    if not images:
        print("❌ A 文件夹中没有图像！")
        return

    print(f"✅ 找到 {len(images)} 张图，将每秒复制一张到 B 文件夹中...")

    index = 0
    while True:
        src_path = images[index]
        dst_path = os.path.join(B_DIR, TARGET_FILENAME)

        # 先尝试删除旧文件
        if os.path.exists(dst_path):
            try:
                os.remove(dst_path)
                print(f"🗑️ 已删除旧图像: {TARGET_FILENAME}")
            except Exception as e:
                print(f"⚠️ 删除失败：{e}")

        # 再复制新文件
        try:
            shutil.copy(src_path, dst_path)
            print(f"🖼️ 复制第 {index+1}/{len(images)} 张图: {os.path.basename(src_path)} → {TARGET_FILENAME}")
        except Exception as e:
            print(f"⚠️ 复制失败：{e}")

        index = (index + 1) % len(images)
        time.sleep(1)

if __name__ == "__main__":
    loop_copy_images()