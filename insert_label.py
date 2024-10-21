import cv2
import os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from PIL.ExifTags import TAGS
import numpy as np

def get_exif_data(image_path):
    image = Image.open(image_path)
    exif_data = image._getexif()
    if exif_data is not None:
        exif = {TAGS[k]: v for k, v in exif_data.items() if k in TAGS}
        return exif
    return {}

def get_capture_date(exif_data):
    date_str = exif_data.get('DateTimeOriginal', None)
    if date_str:
        return datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
    return None

def add_text_to_image(img, text):
    # PILを使ってテキストを描画
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default()
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
    
    # テキストの開始位置（左上の座標）
    text_start_x = 20
    text_start_y = 40
    
    # 矩形の左上と右下の座標（テキストの周りに10pxの余白を追加）
    top_left = (text_start_x - 10, text_start_y - 10)
    bottom_right = (text_start_x + text_size[0] + 10, text_start_y + text_size[1] + 10)
    
    # 矩形を描画
    draw.rectangle([top_left, bottom_right], fill="white")
    
    # 文字列を画像に描画
    draw.text((text_start_x, text_start_y), text, fill="black", font=font)
    
    # OpenCV形式に変換して返す
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def create_fade_frames(image, num_frames=10, fade_in=True):
    frames = []
    alpha_values = np.linspace(0, 1, num_frames) if fade_in else np.linspace(1, 0, num_frames)
    for alpha in alpha_values:
        overlay = image.copy()
        output = image.copy()
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        frames.append(output)
    return frames

# 現在のディレクトリを取得
current_dir = os.path.dirname(os.path.abspath(__file__))

# JPGファイルを取得
jpg_files = [f for f in os.listdir(current_dir) if f.endswith('.JPG')]

# 基準日
base_date = datetime(2024, 5, 3)

# 保存した画像のリスト
saved_images = []

for jpg_file in jpg_files:
    # 画像を読み込む
    img = cv2.imread(os.path.join(current_dir, jpg_file))
    
    # ファイル名
    file_name = jpg_file

    # 撮影日時を取得
    file_path = os.path.join(current_dir, jpg_file)
    exif_data = get_exif_data(file_path)
    capture_date = get_capture_date(exif_data)
    
    if capture_date is None:
        continue  # 撮影日時が取得できない場合はスキップ
    
    # 撮影日時の曜日
    capture_weekday = capture_date.strftime('%A')
    
    # 2024/05/03 起点からの経過日数
    days_elapsed = (capture_date - base_date).days
    
    # 描画する文字列
    text = f"File: {file_name} Date: {capture_date} {capture_weekday} / {days_elapsed} passed days"
    
    # 文字列を画像に描画
    img = add_text_to_image(img, text)
    
    # 画像を保存
    output_path = os.path.join(current_dir, f"annotated_{file_name}")
    cv2.imwrite(output_path, img)
    
    # 保存した画像をリストに追加
    saved_images.append(output_path)

# GIFファイルを作成
frames = []
durations = []

for image_path in saved_images:
    image = Image.open(image_path)
    fade_in_frames = create_fade_frames(np.array(image), num_frames=10, fade_in=True)
    fade_out_frames = create_fade_frames(np.array(image), num_frames=10, fade_in=False)
    
    for frame in fade_in_frames:
        frames.append(Image.fromarray(frame))
        durations.append(100)  # 0.1秒
    
    frames.append(image)
    durations.append(1500)  # 1.5秒
    
    for frame in fade_out_frames:
        frames.append(Image.fromarray(frame))
        durations.append(100)  # 0.1秒

# GIFを保存
frames[0].save(
    os.path.join(current_dir, 'output.gif'),
    save_all=True,
    append_images=frames[1:],
    duration=durations,
    loop=0
)
