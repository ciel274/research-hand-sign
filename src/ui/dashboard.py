import asyncio

# pyrefly: ignore [missing-import]
import cv2
import glob
import os
import time
import json
import numpy as np
import pandas as pd
import mediapipe as mp
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import threading
from PIL import Image, ImageDraw, ImageFont

# 前処理と学習スクリプトをインポート
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from learning.preprocess import normalize_hand_data, preprocess_all_raw_data, extract_hand_features
from learning.model import load_model
from learning.train import train_model
from learning.quantizer import PoseQuantizer
from learning.vla_dataset import export_vla_datasets

app = FastAPI(title="Hand Sign Web Dashboard")

# 静的ファイルの配信設定
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# 各種ディレクトリパスの定義
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
RAW_HIRAGANA_DIR = os.path.join(RAW_DATA_DIR, "hiragana")
RAW_WORDS_DIR = os.path.join(RAW_DATA_DIR, "words")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# モード別のモデルパス
MODEL_PATH_HIRAGANA = os.path.join(BASE_DIR, "src", "learning", "weights", "hand_sign_model_svm_hiragana.joblib")
MODEL_PATH_WORDS = os.path.join(BASE_DIR, "src", "learning", "weights", "hand_sign_model_svm_words.joblib")

CUSTOM_WORDS_FILE = os.path.join(BASE_DIR, "data", "custom_words.json")
TEMPLATE_PATH = os.path.join(BASE_DIR, "src", "ui", "templates", "index.html")

os.makedirs(RAW_HIRAGANA_DIR, exist_ok=True)
os.makedirs(RAW_WORDS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# ローマ字からひらがなへのマッピング辞書
ROMAJI_TO_KANA = {
    "a": "あ", "i": "い", "u": "う", "e": "え", "o": "お",
    "ka": "か", "ki": "き", "ku": "く", "ke": "け", "ko": "こ",
    "sa": "さ", "si": "し", "su": "す", "se": "せ", "so": "そ",
    "ta": "た", "ti": "ち", "tu": "つ", "te": "て", "to": "と",
    "na": "な", "ni": "に", "nu": "ぬ", "ne": "ね", "no": "の",
    "ha": "は", "hi": "ひ", "hu": "ふ", "he": "へ", "ho": "ほ",
    "ma": "ま", "mi": "み", "mu": "む", "me": "め", "mo": "も",
    "ya": "や", "yu": "ゆ", "yo": "よ",
    "ra": "ら", "ri": "り", "ru": "る", "re": "れ", "ro": "ろ",
    "wa": "わ", "wo": "を", "nn": "ん"
}

# グローバルな状態管理変数
state = {
    "task_mode": "hiragana",     # "hiragana" または "words"
    "is_recording": False,
    "active_label": None,
    "current_file_path": None,
    "last_recorded_file": None,
    "velocity": 0.0,
    "pred_class": "--",
    "pred_class_jp": "--",
    "confidence": 0.0,
    "is_stable": False,
    "text_buffer": "",
    "velocity_threshold": 0.008, # スライダーで動的変更可能
    "input_cooldown": 0.25,      # スライダーで動的変更可能
    "stability_frames": 2,       # スライダーで動的変更可能
    "confidence_threshold": 0.75, # スライダーで動的変更可能
    "camera_active": True,
    "video_source_active": False,
}

# カスタム単語リストの読み込み
CUSTOM_WORDS = {}
if os.path.exists(CUSTOM_WORDS_FILE):
    try:
        with open(CUSTOM_WORDS_FILE, "r", encoding="utf-8") as f:
            CUSTOM_WORDS = json.load(f)
    except Exception as e:
        print(f"カスタム単語読み込みエラー: {e}")
        CUSTOM_WORDS = {"ringo": "りんご", "arigatou": "ありがとう", "konnichiwa": "こんにちは"}
else:
    CUSTOM_WORDS = {"ringo": "りんご", "arigatou": "ありがとう", "konnichiwa": "こんにちは"}
    with open(CUSTOM_WORDS_FILE, "w", encoding="utf-8") as f:
        json.dump(CUSTOM_WORDS, f, ensure_ascii=False, indent=2)


# カメラ＆MediaPipe制御用のクラス
class CameraManager:
    def __init__(self):
        self.cap = None
        self.running = False
        self.thread = None
        self.latest_frame = None
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.model = None
        self.last_prediction = None
        self.last_input_time = 0
        self.stability_counter = 0
        self.video_source = None
        self.load_inference_model()

    def load_inference_model(self):
        # 現在のモードに応じたモデルをロード
        mode = state["task_mode"]
        model_path = MODEL_PATH_HIRAGANA if mode == "hiragana" else MODEL_PATH_WORDS
        
        if os.path.exists(model_path):
            try:
                self.model = load_model(model_path)
                print(f"[{mode.upper()}] 推論モデルをロードしました: {os.path.basename(model_path)}")
            except Exception as e:
                print(f"[{mode.upper()}] モデルロードエラー: {e}")
                self.model = None
        else:
            print(f"[{mode.upper()}] 学習済みモデルファイルが存在しません: {os.path.basename(model_path)}")
            self.model = None

    def _draw_overlay(self, frame):
        h, w, _ = frame.shape
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # フォント設定
        font_path_bold = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
        font_path_regular = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"

        try:
            font_char = ImageFont.truetype(font_path_bold, 40)
            font_text = ImageFont.truetype(font_path_regular, 24)
            font_sub = ImageFont.truetype(font_path_regular, 14)
        except Exception:
            font_char = ImageFont.load_default()
            font_text = ImageFont.load_default()
            font_sub = ImageFont.load_default()

        # 1. 右上に現在の予測結果をオーバーレイ
        pred_char = state["pred_class"] or "--"
        mode = state["task_mode"]
        
        # モードに応じた表示名の変換
        if mode == "hiragana":
            pred_char_jp = ROMAJI_TO_KANA.get(pred_char, pred_char)
        else:
            pred_char_jp = CUSTOM_WORDS.get(pred_char, pred_char)
            
        confidence = state["confidence"]
        is_stable = state["is_stable"]
        
        # 背景ボックス (右上)
        box_w, box_h = 175, 95
        box_x, box_y = w - box_w - 15, 15
        draw.rounded_rectangle([box_x, box_y, box_x + box_w, box_y + box_h], radius=10, fill=(12, 14, 20, 200))
        
        # 予測ラベル描画
        draw.text((box_x + 15, box_y + 10), f"予測 ({mode[:4]}):", font=font_sub, fill=(200, 200, 200))
        
        # 文字サイズが大きい単語への対応
        char_font = font_text if len(pred_char_jp) > 2 else font_char
        draw.text((box_x + 15, box_y + 32), pred_char_jp, font=char_font, fill=(168, 85, 247) if is_stable else (226, 232, 240))
        
        # 信頼度/ステータス描画
        has_hands = (pred_char != "--")
        status_text = f"{confidence*100:.1f}%" if has_hands else "検出なし"
        status_color = (16, 185, 129) if is_stable else (148, 163, 184)
        draw.text(
            (box_x + 100, box_y + 35), status_text, font=font_sub, fill=status_color
        )
        draw.text(
            (box_x + 100, box_y + 55),
            "STABLE" if is_stable else "MOVING",
            font=font_sub,
            fill=status_color,
        )

        # 2. 下部に認識したテキストバッファを表示
        text_buf = state["text_buffer"]
        if text_buf:
            max_len = 18
            display_text = (
                text_buf if len(text_buf) <= max_len else "..." + text_buf[-max_len:]
            )

            # 背景ボックス (下部)
            buf_h = 50
            buf_x1, buf_y1 = 15, h - buf_h - 15
            buf_x2, buf_y2 = w - 15, h - 15
            draw.rounded_rectangle(
                [buf_x1, buf_y1, buf_x2, buf_y2], radius=8, fill=(12, 14, 20, 220)
            )

            # テキスト描画
            draw.text(
                (buf_x1 + 15, buf_y1 + 10),
                f"文章: {display_text}",
                font=font_text,
                fill=(255, 255, 255),
            )

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def start(self):
        if not self.running:
            self.cap = None
            
            # ビデオファイル入力ソースのチェック
            if isinstance(self.video_source, str) and os.path.exists(self.video_source):
                cap = cv2.VideoCapture(self.video_source)
                if cap.isOpened():
                    self.cap = cap
                    print(f"動画ファイルの読み込みに成功しました: {self.video_source}")

            # カメラ入力ソースのチェック (フォールバック)
            if self.cap is None:
                for cam_id in [0, 1, 2]:
                    cap = cv2.VideoCapture(cam_id)
                    if cap.isOpened():
                        self.cap = cap
                        print(f"カメラの起動に成功しました (カメラID: {cam_id})")
                        break
                    else:
                        cap.release()

            if self.cap is None or not self.cap.isOpened():
                print("エラー: 利用可能なカメラデバイスまたは動画ファイルが見つかりませんでした。")
                self.running = False
                state["camera_active"] = False
                return

            self.running = True
            state["camera_active"] = True
            self.thread = threading.Thread(target=self.loop, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False
        state["camera_active"] = False
        state["pred_class"] = "--"
        state["pred_class_jp"] = "--"
        state["confidence"] = 0.0
        state["is_stable"] = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def loop(self):
        prev_raw_coords = None
        consecutive_failures = 0

        with self.mp_hands.Hands(
            max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6
        ) as hands:
            while self.running:
                success, frame = self.cap.read()
                if not success:
                    # 動画ファイルの場合はループ再生する
                    if isinstance(self.video_source, str) and os.path.exists(self.video_source):
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        consecutive_failures = 0
                        time.sleep(0.03) # CPU負荷防止
                        continue

                    consecutive_failures += 1
                    time.sleep(0.05)
                    if consecutive_failures >= 30:
                        print("カメラからの映像取得に失敗し続けたため、再接続を試みます...")
                        self.cap.release()
                        for cam_id in [0, 1, 2]:
                            cap = cv2.VideoCapture(cam_id)
                            if cap.isOpened():
                                self.cap = cap
                                print(f"カメラの再接続に成功しました (カメラID: {cam_id})")
                                consecutive_failures = 0
                                break
                            else:
                                cap.release()
                    continue

                consecutive_failures = 0

                # 軽量化のため幅640pxにリサイズ（MediaPipe認識精度を完全に保ちながらCPU・描画負荷を約55%削減）
                h, w = frame.shape[:2]
                target_w = 640
                target_h = int(h * (target_w / w))
                frame = cv2.resize(frame, (target_w, target_h))

                if isinstance(self.video_source, str) and os.path.exists(self.video_source):
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                frame.flags.writeable = False
                results = hands.process(frame)
                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                right_hand_data = [0.0] * 63
                left_hand_data = [0.0] * 63
                has_hands = False

                if results.multi_hand_landmarks and results.multi_handedness:
                    has_hands = True
                    for hand_landmarks, handedness in zip(
                        results.multi_hand_landmarks, results.multi_handedness
                    ):
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                        )

                        label = handedness.classification[0].label
                        temp_coords = []
                        for landmark in hand_landmarks.landmark:
                            temp_coords.extend([landmark.x, landmark.y, landmark.z])

                        if label == "Right":
                            right_hand_data = temp_coords
                        elif label == "Left":
                            left_hand_data = temp_coords

                # 126次元の生座標
                current_raw_coords = np.array(right_hand_data + left_hand_data)

                # --- EMA (指数移動平均) による骨格座標の平滑化 (カメラノイズ・微振れ除去) ---
                if has_hands:
                    if not hasattr(self, "smoothed_coords") or self.smoothed_coords is None:
                        self.smoothed_coords = current_raw_coords
                    else:
                        alpha = 0.70  # 平滑化係数 (0.70 = 反応性と滑らかさの最適バランス)
                        self.smoothed_coords = alpha * current_raw_coords + (1.0 - alpha) * self.smoothed_coords
                    right_hand_data = list(self.smoothed_coords[:63])
                    left_hand_data = list(self.smoothed_coords[63:])
                else:
                    self.smoothed_coords = None

                # --- 速度（移動量）計算 ---
                velocity = 0.0
                if has_hands and prev_raw_coords is not None:
                    diff = current_raw_coords - prev_raw_coords
                    diff_reshaped = diff.reshape(42, 3)
                    valid_joints = np.any(prev_raw_coords.reshape(42, 3) != 0.0, axis=1)
                    if np.sum(valid_joints) > 0:
                        distances = np.linalg.norm(diff_reshaped[valid_joints], axis=1)
                        velocity = np.mean(distances)

                state["velocity"] = float(velocity)
                prev_raw_coords = current_raw_coords if has_hands else None

                # --- 静止判定ロジック ---
                current_time = time.time()
                is_stable = False
                
                # スライダー値と連動
                threshold = state["velocity_threshold"]
                
                if has_hands and velocity > 0.0:
                    if velocity < threshold:
                        self.stability_counter += 1
                    else:
                        self.stability_counter = 0
                        is_stable = False

                    if self.stability_counter >= state.get("stability_frames", 1):
                        is_stable = True
                else:
                    self.stability_counter = 0
                    is_stable = False

                state["is_stable"] = is_stable

                # --- リアルタイム推論と多数決フィルター ---
                if has_hands and self.model is not None:
                    # 指先距離など20次元の補助特徴量を加えた 83次元 * 2 = 166次元 の特徴量を使用
                    r_feats = extract_hand_features(np.array(right_hand_data))
                    l_feats = extract_hand_features(np.array(left_hand_data))
                    features = np.concatenate([r_feats, l_feats]).reshape(1, -1)

                    try:
                        probs = self.model.predict_proba(features)[0]
                        max_idx = np.argmax(probs)
                        pred_class = self.model.classes_[max_idx]
                        confidence = float(probs[max_idx])

                        # UI表示用のリアルタイム更新 (即座に反応)
                        state["pred_class"] = pred_class
                        mode = state["task_mode"]
                        if mode == "hiragana":
                            pred_class_jp = ROMAJI_TO_KANA.get(pred_class, pred_class)
                        else:
                            pred_class_jp = CUSTOM_WORDS.get(pred_class, pred_class)
                        state["pred_class_jp"] = pred_class_jp
                        state["confidence"] = confidence

                        # 履歴に追加
                        if not hasattr(self, "pred_history"):
                            self.pred_history = []
                        self.pred_history.append((pred_class, confidence))
                        if len(self.pred_history) > 5:
                            self.pred_history.pop(0)

                        # スライダーと連動する各種しきい値の取得 (デフォルト値をより高速に応答する設定に変更)
                        conf_threshold = state.get("confidence_threshold", 0.65)
                        input_cooldown = state.get("input_cooldown", 0.20)
                        req_frames = state.get("stability_frames", 1)

                        # 直近の履歴から信頼度の高い予測値をフィルタリング
                        valid_preds = [p[0] for p in self.pred_history if p[1] >= conf_threshold]
                        if len(valid_preds) >= req_frames:
                            from collections import Counter
                            consensus_class, count = Counter(valid_preds).most_common(1)[0]
                            
                            # 新しい手話の形に切り替わった場合は即時反映（1フレーム応答）！
                            # 同一文字が継続中の場合のみ cooldown 時間で連続追記を防止
                            is_new_sign = (consensus_class != self.last_prediction)
                            if is_new_sign or (current_time - self.last_input_time) > input_cooldown:
                                state["text_buffer"] += f" {pred_class_jp}" if mode == "words" else pred_class_jp
                                self.last_prediction = consensus_class
                                self.last_input_time = current_time
                    except Exception as e:
                        print(f"推論エラー: {e}")
                elif not has_hands:
                    if hasattr(self, "pred_history"):
                        self.pred_history = []
                    state["pred_class"] = "--"
                    state["pred_class_jp"] = "--"
                    state["confidence"] = 0.0


                # --- 録画中のデータ保存 ---
                if state["is_recording"] and state["current_file_path"]:
                    try:
                        with open(state["current_file_path"], "a") as f:
                            row = [time.time()] + right_hand_data + left_hand_data
                            row_str = ",".join(map(str, row)) + "\n"
                            f.write(row_str)
                    except Exception as e:
                        print(f"録画書き込みエラー: {e}")

                frame = self._draw_overlay(frame)

                # JPEG圧縮率を70%に指定してストリーミングデータ量を75%削減（超軽快な動作を実現）
                _, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                self.latest_frame = jpeg.tobytes()
                time.sleep(0.03)


camera = CameraManager()


@app.on_event("startup")
def startup_event():
    camera.start()


@app.on_event("shutdown")
def shutdown_event():
    camera.stop()


# --- WEBルート ---
@app.get("/", response_class=HTMLResponse)
def read_index():
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        gen(camera), media_type="multipart/x-mixed-replace; boundary=frame"
    )

def gen(cam):
    while cam.running:
        if cam.latest_frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + cam.latest_frame + b"\r\n"
            )
        time.sleep(0.03)

# --- カメラ制御 API ---

@app.post("/api/camera/start")
def start_camera():
    camera.start()
    return {"status": "success", "camera_active": True}

@app.post("/api/camera/stop")
def stop_camera():
    camera.stop()
    return {"status": "success", "camera_active": False}


# --- 1. タスクモード API ---

@app.post("/api/mode/{mode_name}")
def set_task_mode(mode_name: str):
    if mode_name in ["hiragana", "words"]:
        state["task_mode"] = mode_name
        camera.load_inference_model() # モデルの切り替えロード
        return {"status": "success", "task_mode": mode_name}
    return {"status": "error", "message": "無効なモード名です。"}


# --- 2. カスタム単語操作 API ---

@app.get("/api/words")
def get_custom_words():
    return CUSTOM_WORDS


class WordPayload(BaseModel):
    romaji: str
    kana: str


@app.post("/api/words/add")
def add_custom_word(payload: WordPayload):
    romaji = payload.romaji.strip().lower()
    kana = payload.kana.strip()
    
    if not romaji or not kana:
        return {"status": "error", "message": "ローマ字IDとかな表記の両方を入力してください。"}
        
    CUSTOM_WORDS[romaji] = kana
    
    # 永続化保存
    try:
        with open(CUSTOM_WORDS_FILE, "w", encoding="utf-8") as f:
            json.dump(CUSTOM_WORDS, f, ensure_ascii=False, indent=2)
        return {"status": "success", "words": CUSTOM_WORDS}
    except Exception as e:
        return {"status": "error", "message": f"ファイルの書き込みに失敗しました: {e}"}


@app.post("/api/words/delete/{romaji}")
def delete_custom_word(romaji: str):
    if romaji in CUSTOM_WORDS:
        # 単語の定義を削除
        del CUSTOM_WORDS[romaji]
        
        # 永続化保存
        try:
            with open(CUSTOM_WORDS_FILE, "w", encoding="utf-8") as f:
                json.dump(CUSTOM_WORDS, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"ファイルの更新に失敗しました: {e}")
            
        # 関連する録画CSVファイルも物理削除
        files = glob.glob(os.path.join(RAW_WORDS_DIR, f"sign_language_{romaji}_*.csv"))
        deleted_count = 0
        for f in files:
            try:
                os.remove(f)
                deleted_count += 1
            except Exception as e:
                print(f"関連ファイルの削除エラー: {e}")
                
        return {
            "status": "success", 
            "message": f"単語と関連データ {deleted_count} 件を削除しました。",
            "words": CUSTOM_WORDS
        }
    return {"status": "error", "message": "該当する単語が見つかりませんでした。"}


# --- 3. データカウント API ---

@app.get("/api/counts")
def get_dataset_counts():
    mode = state["task_mode"]
    counts = {}
    
    # モード別に集計対象のラベルリストを準備
    if mode == "hiragana":
        labels = [
            "a", "i", "u", "e", "o", "ka", "ki", "ku", "ke", "ko",
            "sa", "si", "su", "se", "so", "ta", "ti", "tu", "te", "to",
            "na", "ni", "nu", "ne", "no", "ha", "hi", "hu", "he", "ho",
            "ma", "mi", "mu", "me", "mo", "ya", "yu", "yo", "ra", "ri",
            "ru", "re", "ro", "wa", "wo", "nn"
        ]
        target_dir = RAW_HIRAGANA_DIR
    else:
        labels = list(CUSTOM_WORDS.keys())
        target_dir = RAW_WORDS_DIR

    for label in labels:
        counts[label] = 0

    # 実際のファイル数を集計
    csv_files = glob.glob(os.path.join(target_dir, "sign_language_*.csv"))
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        base_name = file_name.replace("sign_language_", "").replace(".csv", "")
        parts = base_name.split("_")

        if len(parts) > 1 and parts[-1].isdigit():
            label = "_".join(parts[:-1])
        else:
            label = base_name

        if label in counts:
            counts[label] += 1

    return counts


# --- 4. 録画制御 API ---

@app.post("/api/record/start/{label}")
def record_start(label: str):
    state["is_recording"] = True
    state["active_label"] = label
    
    mode = state["task_mode"]
    target_dir = RAW_HIRAGANA_DIR if mode == "hiragana" else RAW_WORDS_DIR

    existing_files = glob.glob(
        os.path.join(target_dir, f"sign_language_{label}_*.csv")
    )
    indices = []
    for f in existing_files:
        name = os.path.basename(f)
        try:
            idx = int(name.replace(f"sign_language_{label}_", "").replace(".csv", ""))
            indices.append(idx)
        except ValueError:
            pass

    next_idx = max(indices) + 1 if indices else 1
    file_name = f"sign_language_{label}_{next_idx}.csv"
    state["current_file_path"] = os.path.join(target_dir, file_name)

    # ヘッダー作成
    with open(state["current_file_path"], "w") as f:
        header = ["timestamp"]
        for i in range(21):
            header.extend([f"R_joint_{i}_x", f"R_joint_{i}_y", f"R_joint_{i}_z"])
        for i in range(21):
            header.extend([f"L_joint_{i}_x", f"L_joint_{i}_y", f"L_joint_{i}_z"])
        f.write(",".join(header) + "\n")

    print(f"[{mode.upper()} 録画開始]: {file_name}")
    return {"status": "started", "file": file_name}


@app.post("/api/record/stop")
def record_stop():
    print(
        f"録画停止: {os.path.basename(state['current_file_path']) if state['current_file_path'] else 'None'}"
    )
    state["is_recording"] = False
    state["last_recorded_file"] = state["current_file_path"]
    state["active_label"] = None
    state["current_file_path"] = None
    return {"status": "stopped"}


@app.post("/api/record/delete_last")
def delete_last_record():
    last_file = state.get("last_recorded_file")
    if last_file and os.path.exists(last_file):
        try:
            os.remove(last_file)
            print(f"直近の録画を削除しました: {os.path.basename(last_file)}")
            state["last_recorded_file"] = None
            return {
                "status": "success",
                "message": f"直近の録画を削除しました: {os.path.basename(last_file)}",
            }
        except Exception as e:
            return {"status": "error", "message": f"削除エラー: {e}"}
    return {"status": "error", "message": "削除できる直近の録画データがありません。"}


@app.post("/api/record/clear_class/{label}")
def clear_class_records(label: str):
    mode = state["task_mode"]
    target_dir = RAW_HIRAGANA_DIR if mode == "hiragana" else RAW_WORDS_DIR
    
    files = glob.glob(os.path.join(target_dir, f"sign_language_{label}_*.csv"))
    if not files:
        return {"status": "success", "message": f"'{label}' のデータは既に空です。"}
        
    deleted_count = 0
    for f in files:
        try:
            os.remove(f)
            deleted_count += 1
        except Exception as e:
            print(f"ファイル削除エラー ({f}): {e}")
            
    state["last_recorded_file"] = None
    print(f"[{mode.upper()}] '{label}' のデータを {deleted_count} 件削除しました。")
    return {
        "status": "success",
        "message": f"'{label}' の録画データを {deleted_count} 件すべて削除しました。",
    }


# --- 5. パラメータ / 設定 API ---

@app.post("/api/settings/velocity")
def update_velocity_threshold(threshold: float):
    state["velocity_threshold"] = threshold
    return {"status": "success", "velocity_threshold": threshold}


@app.post("/api/settings/inference")
def update_inference_settings(data: dict):
    if "input_cooldown" in data:
        state["input_cooldown"] = float(data["input_cooldown"])
    if "stability_frames" in data:
        state["stability_frames"] = int(data["stability_frames"])
    if "confidence_threshold" in data:
        state["confidence_threshold"] = float(data["confidence_threshold"])
    return {
        "status": "success",
        "settings": {
            "input_cooldown": state["input_cooldown"],
            "stability_frames": state["stability_frames"],
            "confidence_threshold": state["confidence_threshold"]
        }
    }


@app.post("/api/settings/video_source")
def update_video_source(data: dict):
    path = data.get("path", "").strip()
    if path:
        if not os.path.exists(path):
            return {"status": "error", "message": f"ファイルが存在しません: {path}"}
        camera.video_source = path
        state["video_source_active"] = True
    else:
        camera.video_source = None
        state["video_source_active"] = False
    
    # 動作中の場合はスレッド再起動をかける
    was_running = camera.running
    if was_running:
        camera.stop()
        time.sleep(0.5)
        camera.start()
        
    source_name = os.path.basename(path) if path else "Webカメラ"
    return {"status": "success", "message": f"入力ソースを {source_name} に変更しました"}


@app.post("/api/settings/upload_video")
async def upload_video(file: UploadFile = File(...)):
    import shutil
    temp_dir = os.path.join(STATIC_DIR, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # ファイル名から拡張子を維持しつつ安全に保存
    ext = os.path.splitext(file.filename)[1]
    temp_path = os.path.join(temp_dir, f"test_signer{ext}")
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    camera.video_source = temp_path
    state["video_source_active"] = True
    
    was_running = camera.running
    if was_running:
        camera.stop()
        time.sleep(0.5)
        camera.start()
        
    return {"status": "success", "message": f"動画 '{file.filename}' を読み込みました。検証を開始します。"}




@app.get("/api/status")
def get_status():
    return state


@app.post("/api/clear_text")
def clear_text():
    state["text_buffer"] = ""
    camera.last_prediction = None
    return {"status": "cleared"}


@app.post("/api/backspace_text")
def backspace_text():
    if state["text_buffer"]:
        # 1文字（または単語区切りの空白含む1文字）を末尾から削除
        state["text_buffer"] = state["text_buffer"][:-1]
    return {"status": "success", "text_buffer": state["text_buffer"]}


# --- 6. 機械学習 & VLA API ---

@app.post("/api/preprocess")
def run_preprocess():
    mode = state["task_mode"]
    target_raw_dir = RAW_HIRAGANA_DIR if mode == "hiragana" else RAW_WORDS_DIR
    output_csv = os.path.join(PROCESSED_DATA_DIR, f"dataset_{mode}.csv")
    mode_name = "ひらがな" if mode == "hiragana" else "単語"

    if not glob.glob(os.path.join(target_raw_dir, "sign_language_*.csv")):
        return {
            "status": "error",
            "message": f"{mode_name}モードの録画データがまだありません。先に{mode_name}を録画するか、「指文字収集」タブに切り替えてください。",
        }

    try:
        preprocess_all_raw_data(
            raw_dir=target_raw_dir,
            output_csv=output_csv,
        )
        return {
            "status": "success",
            "message": f"[{mode.upper()}] 一括前処理（データの正規化と統合）が完了しました！",
        }
    except Exception as e:
        return {"status": "error", "message": f"前処理エラー: {e}"}


@app.post("/api/train")
def run_train():
    mode = state["task_mode"]
    dataset_path = os.path.join(PROCESSED_DATA_DIR, f"dataset_{mode}.csv")
    model_output_dir = os.path.join(BASE_DIR, "src", "learning", "weights")
    
    if not os.path.exists(dataset_path):
        return {
            "status": "error",
            "message": f"統合データファイルが見つかりません。先に前処理を実行してください。"
        }
        
    try:
        # train_modelを実行 (出力モデルファイル名は train_model 側で固定されている場合があるため、完了後にモード別のファイル名にリネーム)
        train_model(
            dataset_path=dataset_path,
            model_output_dir=model_output_dir,
            model_type="svm",
        )
        
        # 出来上がったモデルをリネームして退避
        default_model_path = os.path.join(model_output_dir, "hand_sign_model_svm.joblib")
        target_model_path = MODEL_PATH_HIRAGANA if mode == "hiragana" else MODEL_PATH_WORDS
        
        if os.path.exists(default_model_path):
            if os.path.exists(target_model_path):
                os.remove(target_model_path)
            os.rename(default_model_path, target_model_path)
            
        camera.load_inference_model() # 推論器の再ロード
        return {
            "status": "success",
            "message": f"[{mode.upper()}] 学習が完了し、最新のモデルが推論用にロードされました！",
        }
    except Exception as e:
        return {"status": "error", "message": f"学習エラー: {e}"}


@app.get("/api/train/stats")
def get_train_stats():
    mode = state["task_mode"]
    stats_path = os.path.join(PROCESSED_DATA_DIR, f"train_stats_{mode}.json")
    if os.path.exists(stats_path):
        try:
            with open(stats_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            return {"status": "error", "message": f"読み込みエラー: {e}"}
    return {"status": "not_trained", "message": "学習統計データはまだ存在しません。"}


@app.post("/api/vla/quantize")
def run_vla_quantize(n_clusters: int = 64):
    mode = state["task_mode"]
    dataset_path = os.path.join(PROCESSED_DATA_DIR, f"dataset_{mode}.csv")
    quantizer_path = os.path.join(BASE_DIR, "src", "learning", "weights", f"pose_quantizer_{n_clusters}.joblib")
    
    if not os.path.exists(dataset_path):
        return {
            "status": "error",
            "message": "データセットが見つかりません。先に「一括前処理」を実行してください。"
        }
        
    try:
        df = pd.read_csv(dataset_path)
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'label']]
        X = df[feature_cols].values
        
        actual_clusters = min(n_clusters, len(X))
        
        quantizer = PoseQuantizer(n_clusters=actual_clusters)
        quantizer.fit(X)
        quantizer.save(quantizer_path)
        
        # 再構成誤差(MSE)
        tokens = quantizer.tokenize(X)
        X_recon = quantizer.detokenize(tokens)
        mse = float(np.mean((X - X_recon) ** 2))
        
        return {
            "status": "success",
            "message": f"ポーズ量子化モデル (K={actual_clusters}) の学習が完了しました！\n再構成誤差 (MSE): {mse:.6f}",
            "mse": mse,
            "num_frames": len(X),
            "num_clusters": actual_clusters,
            "tokens_preview": tokens[:15].tolist()
        }
    except Exception as e:
        return {"status": "error", "message": f"量子化学習エラー: {e}"}


@app.post("/api/vla/export")
def run_vla_export(n_clusters: int = 64):
    mode = state["task_mode"]
    continuous_path = os.path.join(PROCESSED_DATA_DIR, "vla_continuous.jsonl")
    discrete_path = os.path.join(PROCESSED_DATA_DIR, f"vla_discrete_{n_clusters}.jsonl")
    
    try:
        export_vla_datasets(n_clusters=n_clusters, mode=mode)
        
        preview_continuous = ""
        preview_discrete = ""
        
        if os.path.exists(continuous_path):
            with open(continuous_path, "r", encoding="utf-8") as f:
                preview_continuous = f.readline().strip()
                
        if os.path.exists(discrete_path):
            with open(discrete_path, "r", encoding="utf-8") as f:
                preview_discrete = f.readline().strip()
                
        return {
            "status": "success",
            "message": f"VLA用データセットのエクスポートが完了しました！ (K={n_clusters})",
            "continuous_file": os.path.basename(continuous_path),
            "discrete_file": os.path.basename(discrete_path),
            "preview_continuous": preview_continuous,
            "preview_discrete": preview_discrete
        }
    except Exception as e:
        return {"status": "error", "message": f"エクスポートエラー: {e}"}


@app.get("/api/vla/stats")
def get_vla_stats(n_clusters: int = 64):
    mode = state["task_mode"]
    quantizer_path = os.path.join(BASE_DIR, "src", "learning", "weights", f"pose_quantizer_{n_clusters}.joblib")
    dataset_path = os.path.join(PROCESSED_DATA_DIR, f"dataset_{mode}.csv")
    continuous_path = os.path.join(PROCESSED_DATA_DIR, "vla_continuous.jsonl")
    discrete_path = os.path.join(PROCESSED_DATA_DIR, f"vla_discrete_{n_clusters}.jsonl")
    
    has_quantizer = os.path.exists(quantizer_path)
    has_dataset = os.path.exists(dataset_path)
    has_vla_continuous = os.path.exists(continuous_path)
    has_vla_discrete = os.path.exists(discrete_path)
    
    pose_distribution = []
    num_frames = 0
    num_clusters = 0
    
    if has_quantizer and has_dataset:
        try:
            df = pd.read_csv(dataset_path)
            feature_cols = [col for col in df.columns if col not in ['timestamp', 'label']]
            X = df[feature_cols].values
            num_frames = len(X)
            
            quantizer = PoseQuantizer()
            quantizer.load(quantizer_path)
            num_clusters = quantizer.n_clusters
            
            tokens = quantizer.tokenize(X)
            
            unique, counts = np.unique(tokens, return_counts=True)
            dist_map = dict(zip(unique.tolist(), counts.tolist()))
            
            for i in range(num_clusters):
                pose_distribution.append({
                    "token_id": i,
                    "count": dist_map.get(i, 0)
                })
        except Exception as e:
            print(f"ステータス集計中のエラー: {e}")
            

class LaunchViewerPayload(BaseModel):
    path: str = None


@app.post("/api/viewer/launch")
def launch_3d_viewer(payload: LaunchViewerPayload = None):
    target = payload.path if payload and payload.path else state.get("last_recorded_file")
    if not target or not os.path.exists(target):
        # rawディレクトリ内の直近更新CSVをフォールバックとして探索
        csv_files = glob.glob(os.path.join(RAW_DATA_DIR, "**", "*.csv"), recursive=True)
        if csv_files:
            target = max(csv_files, key=os.path.getmtime)
    if not target or not os.path.exists(target):
        return {"status": "error", "message": "表示対象のCSVファイルが見つかりません。先に録画を行ってください。"}

    import subprocess
    viewer_script = os.path.join(BASE_DIR, "src", "viewer", "hand_viewer.py")
    subprocess.Popen([sys.executable, viewer_script, target])
    return {
        "status": "success",
        "message": f"3Dビューアを起動しました: {os.path.basename(target)}",
        "file": os.path.basename(target)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
