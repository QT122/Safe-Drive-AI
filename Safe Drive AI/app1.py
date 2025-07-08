from flask import Flask, request, jsonify, send_from_directory, send_file
import os
import shutil
import json
from flask_cors import CORS

# 假设你的结果文件路径
UPLOAD_FOLDER = r"D:\python\演示用\.venv\后端\uploads"
RESULT_PATH = r"D:\python\演示用\.venv\后端\result.json"
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# 保存目录
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# 首页
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


# 图像上传
@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image part'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # 清空上传目录
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

        # 保存新文件
        filename = "latest.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        return jsonify({'message': 'File uploaded successfully', 'filename': filename})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 清空图像文件夹
@app.route('/reset', methods=['POST'])
def reset_images():
    try:
        if os.path.exists(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        return jsonify({'message': '图像文件夹已清空'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/result.json')
def serve_result_json():
    headers = {
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0'
    }

    try:
        # 检查文件是否存在
        if not os.path.exists(RESULT_PATH):
            return jsonify({"class_name": "Unknown"}), 200, headers

        # 读取文件内容
        with open(RESULT_PATH, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                return jsonify(data), 200, headers
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}")
                return jsonify({"class_name": "JSON Error", "details": str(e)}), 500, headers
            except Exception as e:
                print(f"读取文件内容错误: {e}")
                return jsonify({"class_name": "Error"}), 500, headers

    except Exception as e:
        print(f"处理结果请求时发生未预期错误: {e}")
        return jsonify({"class_name": "Unknown", "error": str(e)}), 500, headers

#评分
# 扣分详情页面（HTML）
@app.route('/score_page')
def score_page():
    return send_from_directory('static', 'score_page.html')  # 注意：你要把HTML放在 static 文件夹里

# 扣分数据接口
@app.route('/score_data')
def score_data():
    log_path = os.path.join(os.path.dirname(__file__), "violation_log.txt")
    if not os.path.exists(log_path):
        return jsonify({"score": 100, "violations": []})

    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    filtered = []
    last_type = None

    for line in lines:
        if "Detected" not in line:
            continue

        for keyword in ["closing of eyes", "yawning", "object", "extra hand", "seat belt was not fastened", "steering wheel"]:
            if keyword in line:
                if keyword != last_type:
                    filtered.append(line.strip())
                    last_type = keyword
                break

    # 只根据唯一行为种类来扣分
    unique_types = set()
    for v in filtered:
        for k in ["closing of eyes", "yawning", "object", "extra hand", "seat belt was not fastened", "steering wheel"]:
            if k in v:
                unique_types.add(k)
                break

    score = max(0, 100 - len(filtered))

    return jsonify({
        "score": score,
        "violations": filtered
    })



if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)
