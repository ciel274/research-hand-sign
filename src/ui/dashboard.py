import asyncio

# pyrefly: ignore [missing-import]
import cv2
import glob
import os
import time
import numpy as np
import mediapipe as mp
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import threading
from PIL import Image, ImageDraw, ImageFont

# 前処理と学習スクリプトをインポート
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from learning.preprocess import normalize_hand_data, preprocess_all_raw_data
from learning.model import load_model
from learning.train import train_model
from learning.quantizer import PoseQuantizer
from learning.vla_dataset import export_vla_datasets

app = FastAPI(title="Hand Sign Web Dashboard")

# 各種ディレクトリパスの定義
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_PATH = os.path.join(
    BASE_DIR, "src", "learning", "weights", "hand_sign_model_svm.joblib"
)
TEMPLATE_PATH = os.path.join(BASE_DIR, "src", "ui", "templates", "index.html")

os.makedirs(RAW_DATA_DIR, exist_ok=True)
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
    "is_recording": False,
    "active_label": None,
    "current_file_path": None,
    "last_recorded_file": None,  # 直近の録画ファイルを追跡
    "velocity": 0.0,
    "pred_class": "--",
    "pred_class_jp": "--",
    "confidence": 0.0,
    "is_stable": False,
    "text_buffer": "",
}


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
        self.load_inference_model()
        self.last_prediction = None
        self.last_input_time = 0
        self.stability_counter = 0

    def load_inference_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                self.model = load_model(MODEL_PATH)
                print(f"モデルを読み込みました: {MODEL_PATH}")
            except Exception as e:
                print(f"モデルロードエラー: {e}")
        else:
            self.model = None

    def _draw_overlay(self, frame):
        # PIL Imageに変換して日本語を描画
        h, w, _ = frame.shape
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # フォント設定（macOS環境前提）
        font_path_bold = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
        font_path_regular = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"

        try:
            font_char = ImageFont.truetype(font_path_bold, 48)
            font_text = ImageFont.truetype(font_path_regular, 24)
            font_sub = ImageFont.truetype(font_path_regular, 14)
        except Exception:
            # フォント読み込みエラー時のフォールバック
            font_char = ImageFont.load_default()
            font_text = ImageFont.load_default()
            font_sub = ImageFont.load_default()

        # 1. 右上に現在の予測文字 & 信頼度をオーバーレイ
        pred_char = state["pred_class"] or "--"
        pred_char_jp = ROMAJI_TO_KANA.get(pred_char, pred_char)
        confidence = state["confidence"]
        is_stable = state["is_stable"]
        
        # 背景ボックス (右上)
        box_w, box_h = 160, 90
        box_x, box_y = w - box_w - 15, 15
        draw.rounded_rectangle([box_x, box_y, box_x + box_w, box_y + box_h], radius=10, fill=(12, 14, 20, 200))
        
        # 予測ラベル描画
        draw.text((box_x + 15, box_y + 10), "予測:", font=font_sub, fill=(200, 200, 200))
        draw.text((box_x + 15, box_y + 28), pred_char_jp, font=font_char, fill=(168, 85, 247) if is_stable else (226, 232, 240))
        
        # 信頼度/ステータス描画
        has_hands = (pred_char != "--")
        status_text = f"{confidence*100:.1f}%" if has_hands else "検出なし"
        status_color = (16, 185, 129) if is_stable else (148, 163, 184)
        draw.text(
            (box_x + 85, box_y + 35), status_text, font=font_sub, fill=status_color
        )
        draw.text(
            (box_x + 85, box_y + 55),
            "STABLE" if is_stable else "MOVING",
            font=font_sub,
            fill=status_color,
        )

        # 2. 下部に認識したテキストバッファを表示
        text_buf = state["text_buffer"]
        if text_buf:
            # 文字列が長すぎる場合は末尾を表示
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
            # 利用可能なカメラIDを自動探索 (0, 1, 2)
            self.cap = None
            for cam_id in [0, 1, 2]:
                cap = cv2.VideoCapture(cam_id)
                if cap.isOpened():
                    self.cap = cap
                    print(f"カメラの起動に成功しました (カメラID: {cam_id})")
                    break
                else:
                    cap.release()

            if self.cap is None or not self.cap.isOpened():
                print("エラー: 利用可能なカメラデバイスが見つかりませんでした。")
                self.running = False
                return

            self.running = True
            self.thread = threading.Thread(target=self.loop, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

    def loop(self):
        # 速度計測・安定判定用
        prev_raw_coords = None
        VELOCITY_THRESHOLD = 0.008
        STABILITY_FRAMES = 8
        CONFIDENCE_THRESHOLD = 0.85
        INPUT_COOLDOWN = 1.0

        consecutive_failures = 0

        with self.mp_hands.Hands(
            max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7
        ) as hands:
            while self.running:
                success, frame = self.cap.read()
                if not success:
                    consecutive_failures += 1
                    time.sleep(0.05)
                    # 連続30フレーム(約1.5秒)失敗した場合、カメラ再接続を試みる
                    if consecutive_failures >= 30:
                        print(
                            "カメラからの映像取得に失敗し続けたため、再接続を試みます..."
                        )
                        self.cap.release()
                        for cam_id in [0, 1, 2]:
                            cap = cv2.VideoCapture(cam_id)
                            if cap.isOpened():
                                self.cap = cap
                                print(
                                    f"カメラの再接続に成功しました (カメラID: {cam_id})"
                                )
                                consecutive_failures = 0
                                break
                            else:
                                cap.release()
                    continue

                consecutive_failures = 0

                # 画面反転、RGB変換
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
                if has_hands and velocity > 0.0:
                    if velocity < VELOCITY_THRESHOLD:
                        self.stability_counter += 1
                    else:
                        self.stability_counter = 0
                        is_stable = False

                    if self.stability_counter >= STABILITY_FRAMES:
                        is_stable = True
                else:
                    self.stability_counter = 0
                    is_stable = False

                state["is_stable"] = is_stable

                # --- リアルタイム推論と文字スタック ---
                if is_stable and self.model is not None:
                    r_norm = normalize_hand_data(np.array(right_hand_data))
                    l_norm = normalize_hand_data(np.array(left_hand_data))
                    features = np.concatenate([r_norm, l_norm]).reshape(1, -1)

                    try:
                        probs = self.model.predict_proba(features)[0]
                        max_idx = np.argmax(probs)
                        pred_class = self.model.classes_[max_idx]
                        confidence = probs[max_idx]

                        state["pred_class"] = pred_class
                        state["pred_class_jp"] = ROMAJI_TO_KANA.get(pred_class, pred_class)
                        state["confidence"] = float(confidence)

                        if confidence >= CONFIDENCE_THRESHOLD:
                            if (
                                pred_class != self.last_prediction
                                or (current_time - self.last_input_time)
                                > INPUT_COOLDOWN
                            ):
                                state["text_buffer"] += ROMAJI_TO_KANA.get(pred_class, pred_class)
                                self.last_prediction = pred_class
                                self.last_input_time = current_time
                                self.stability_counter = -5  # しばらく入力をロック
                    except Exception as e:
                        print(f"推論エラー: {e}")
                elif not has_hands:
                    state["pred_class"] = "--"
                    state["pred_class_jp"] = "--"
                    state["confidence"] = 0.0

                # --- 録画中のデータ保存 ---
                if state["is_recording"] and state["current_file_path"]:
                    try:
                        with open(state["current_file_path"], "a") as f:
                            # タイムスタンプ + 右手(63) + 左手(63)
                            row = [time.time()] + right_hand_data + left_hand_data
                            row_str = ",".join(map(str, row)) + "\n"
                            f.write(row_str)
                    except Exception as e:
                        print(f"録画書き込みエラー: {e}")

                # --- フレームに認識結果をオーバーレイ描画 ---
                frame = self._draw_overlay(frame)

                # ストリーム用フレーム更新
                _, jpeg = cv2.imencode(".jpg", frame)
                self.latest_frame = jpeg.tobytes()
                time.sleep(0.03)  # 約30fpsに制限


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


# --- カメラ配信エンドポイント ---
def gen(cam):
    while cam.running:
        if cam.latest_frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + cam.latest_frame + b"\r\n"
            )
        time.sleep(0.03)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        gen(camera), media_type="multipart/x-mixed-replace; boundary=frame"
    )


# --- API エンドポイント ---


@app.get("/api/counts")
def get_dataset_counts():
    """
    data/raw 以下の各文字ごとの収録ファイル数をカウントして返します。
    """
    counts = {}
    # 全ての50音に対して0で初期化
    romaji_list = [
        "a",
        "i",
        "u",
        "e",
        "o",
        "ka",
        "ki",
        "ku",
        "ke",
        "ko",
        "sa",
        "si",
        "su",
        "se",
        "so",
        "ta",
        "ti",
        "tu",
        "te",
        "to",
        "na",
        "ni",
        "nu",
        "ne",
        "no",
        "ha",
        "hi",
        "hu",
        "he",
        "ho",
        "ma",
        "mi",
        "mu",
        "me",
        "mo",
        "ya",
        "yu",
        "yo",
        "ra",
        "ri",
        "ru",
        "re",
        "ro",
        "wa",
        "wo",
        "nn",
    ]
    for ro in romaji_list:
        counts[ro] = 0

    # 実際のファイル数を数える
    csv_files = glob.glob(os.path.join(RAW_DATA_DIR, "sign_language_*.csv"))
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        base_name = file_name.replace("sign_language_", "").replace(".csv", "")
        parts = base_name.split("_")

        # ラベルの特定
        if len(parts) > 1 and parts[-1].isdigit():
            label = "_".join(parts[:-1])
        else:
            label = base_name

        if label in counts:
            counts[label] += 1

    return counts


@app.post("/api/record/start/{label}")
def record_start(label: str):
    """
    録画を開始し、自動ナンバリングした新規CSVファイルを用意します。
    """
    state["is_recording"] = True
    state["active_label"] = label

    # 自動ナンバリング処理
    existing_files = glob.glob(
        os.path.join(RAW_DATA_DIR, f"sign_language_{label}_*.csv")
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
    state["current_file_path"] = os.path.join(RAW_DATA_DIR, file_name)

    # ヘッダーを作成
    with open(state["current_file_path"], "w") as f:
        header = ["timestamp"]
        for i in range(21):
            header.extend([f"R_joint_{i}_x", f"R_joint_{i}_y", f"R_joint_{i}_z"])
        for i in range(21):
            header.extend([f"L_joint_{i}_x", f"L_joint_{i}_y", f"L_joint_{i}_z"])
        f.write(",".join(header) + "\n")

    print(f"録画開始: {file_name}")
    return {"status": "started", "file": file_name}


@app.post("/api/record/stop")
def record_stop():
    """
    録画を停止します。
    """
    print(
        f"録画停止: {os.path.basename(state['current_file_path']) if state['current_file_path'] else 'None'}"
    )
    state["is_recording"] = False
    state["last_recorded_file"] = state["current_file_path"]  # 削除用に記録
    state["active_label"] = None
    state["current_file_path"] = None
    return {"status": "stopped"}


@app.post("/api/record/delete_last")
def delete_last_record():
    """
    直近に録画された1件のCSVファイルを物理削除します。
    """
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
    """
    指定した文字（ラベル）のすべてのCSVデータを物理削除します。
    """
    files = glob.glob(os.path.join(RAW_DATA_DIR, f"sign_language_{label}_*.csv"))
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
    print(f"'{label}' のデータを {deleted_count} 件削除しました。")
    return {
        "status": "success",
        "message": f"'{label}' の録画データを {deleted_count} 件すべて削除しました。",
    }


@app.get("/api/status")
def get_status():
    """
    現在のリアルタイム予測状態や速度、テキストバッファを返します。
    """
    return state


@app.post("/api/clear_text")
def clear_text():
    state["text_buffer"] = ""
    camera.last_prediction = None
    return {"status": "cleared"}


@app.post("/api/preprocess")
def run_preprocess():
    """
    データセットの一括前処理を行います。
    """
    try:
        preprocess_all_raw_data(
            raw_dir=RAW_DATA_DIR,
            output_csv=os.path.join(PROCESSED_DATA_DIR, "dataset.csv"),
        )
        return {
            "status": "success",
            "message": "一括前処理（データの正規化と統合）が完了しました！",
        }
    except Exception as e:
        return {"status": "error", "message": f"前処理エラー: {e}"}


@app.post("/api/train")
def run_train():
    """
    モデルの学習を実行し、完了後に推論用モデルをリロードします。
    """
    try:
        train_model(
            dataset_path=os.path.join(PROCESSED_DATA_DIR, "dataset.csv"),
            model_output_dir=os.path.join(BASE_DIR, "src", "learning", "weights"),
            model_type="svm",
        )
        # 学習が終わったら推論モデルを最新の物にリロード
        camera.load_inference_model()
        return {
            "status": "success",
            "message": "学習が完了し、最新のモデルが推論用にロードされました！",
        }
    except Exception as e:
        return {"status": "error", "message": f"学習エラー: {e}"}


# --- VLA & 離散化関連 API ---

@app.post("/api/vla/quantize")
def run_vla_quantize(n_clusters: int = 64):
    """
    データセットを読み込み、K-Meansでポーズの量子化器を学習させます。
    再構成誤差（MSE）を返します。
    """
    dataset_path = os.path.join(PROCESSED_DATA_DIR, "dataset.csv")
    quantizer_path = os.path.join(BASE_DIR, "src", "learning", "weights", "pose_quantizer.joblib")
    
    if not os.path.exists(dataset_path):
        return {
            "status": "error",
            "message": "データセットが見つかりません。先に「一括前処理」を実行してください。"
        }
        
    try:
        df = pd.read_csv(dataset_path)
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'label']]
        X = df[feature_cols].values
        
        # クラスタ数がデータ数を超えないように調整
        actual_clusters = min(n_clusters, len(X))
        
        quantizer = PoseQuantizer(n_clusters=actual_clusters)
        quantizer.fit(X)
        quantizer.save(quantizer_path)
        
        # 再構成誤差(MSE)の算出
        tokens = quantizer.tokenize(X)
        X_recon = quantizer.detokenize(tokens)
        mse = float(np.mean((X - X_recon) ** 2))
        
        return {
            "status": "success",
            "message": f"ポーズ量子化モデル（K-Means）の学習が完了しました！\nクラスタ数: {actual_clusters}\n再構成誤差 (MSE): {mse:.6f}",
            "mse": mse,
            "num_frames": len(X),
            "num_clusters": actual_clusters,
            "tokens_preview": tokens[:15].tolist()
        }
    except Exception as e:
        return {"status": "error", "message": f"量子化学習エラー: {e}"}


@app.post("/api/vla/export")
def run_vla_export():
    """
    生CSVデータから、アプローチA（連続値）およびB（離散値）のVLA用データセットを書き出します。
    """
    continuous_path = os.path.join(PROCESSED_DATA_DIR, "vla_continuous.jsonl")
    discrete_path = os.path.join(PROCESSED_DATA_DIR, "vla_discrete.jsonl")
    
    try:
        export_vla_datasets()
        
        # エクスポートファイルのプレビューを読み込む
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
            "message": "VLA用データセットの書き出しに成功しました！",
            "continuous_file": os.path.basename(continuous_path),
            "discrete_file": os.path.basename(discrete_path),
            "preview_continuous": preview_continuous,
            "preview_discrete": preview_discrete
        }
    except Exception as e:
        return {"status": "error", "message": f"エクスポートエラー: {e}"}


@app.get("/api/vla/stats")
def get_vla_stats():
    """
    ポーズ量子化器とVLAデータセットの現在のステータス、およびポーズの出現頻度分布を返します。
    """
    quantizer_path = os.path.join(BASE_DIR, "src", "learning", "weights", "pose_quantizer.joblib")
    dataset_path = os.path.join(PROCESSED_DATA_DIR, "dataset.csv")
    continuous_path = os.path.join(PROCESSED_DATA_DIR, "vla_continuous.jsonl")
    discrete_path = os.path.join(PROCESSED_DATA_DIR, "vla_discrete.jsonl")
    
    has_quantizer = os.path.exists(quantizer_path)
    has_dataset = os.path.exists(dataset_path)
    has_vla_continuous = os.path.exists(continuous_path)
    has_vla_discrete = os.path.exists(discrete_path)
    
    # ポーズトークンの頻度集計
    pose_distribution = []
    num_frames = 0
    num_clusters = 0
    
    if has_quantizer and has_dataset:
        try:
            # データをロードしてトークン化
            df = pd.read_csv(dataset_path)
            feature_cols = [col for col in df.columns if col not in ['timestamp', 'label']]
            X = df[feature_cols].values
            num_frames = len(X)
            
            quantizer = PoseQuantizer()
            quantizer.load(quantizer_path)
            num_clusters = quantizer.n_clusters
            
            tokens = quantizer.tokenize(X)
            
            # 各トークンの個数を集計
            unique, counts = np.unique(tokens, return_counts=True)
            dist_map = dict(zip(unique.tolist(), counts.tolist()))
            
            # 全クラスタIDに対する分布リストを作成
            for i in range(num_clusters):
                pose_distribution.append({
                    "token_id": i,
                    "count": dist_map.get(i, 0)
                })
        except Exception as e:
            print(f"ステータス集計中のエラー: {e}")
            
    return {
        "has_quantizer": has_quantizer,
        "has_vla_continuous": has_vla_continuous,
        "has_vla_discrete": has_vla_discrete,
        "num_frames": num_frames,
        "num_clusters": num_clusters,
        "pose_distribution": pose_distribution
    }


if __name__ == "__main__":
    import uvicorn

    # uvicorn dashboard:app --host 0.0.0.0 --port 8000 --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
