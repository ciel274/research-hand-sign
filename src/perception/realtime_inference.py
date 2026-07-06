import cv2
import mediapipe as mp
import numpy as np
import os
import time
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.learning.preprocess import normalize_hand_data
from src.learning.model import load_model

# --- 設定値 ---
MODEL_PATH = "src/learning/weights/hand_sign_model_svm.joblib"
VELOCITY_THRESHOLD = 0.008      # 静止判定の速度しきい値（小さいほど厳密に止める必要がある）
STABILITY_FRAMES = 8           # 静止とみなす継続フレーム数（約0.25秒）
CONFIDENCE_THRESHOLD = 0.85    # 予測の最低信頼度（確率）
INPUT_COOLDOWN = 1.0           # 同じ文字を入力するまでの最低クールダウン時間（秒）

# MediaPipeの初期化
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 学習済みモデルのロード
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        print(f"モデルをロードしました: {MODEL_PATH}")
    except Exception as e:
        print(f"モデルのロードに失敗しました: {e}")
else:
    print(f"警告: モデル {MODEL_PATH} が見つかりません。")
    print("手元のカメラ映像のみを表示します。判定を行うには、まずデータ収集と学習を行ってください。")

# 入力制御変数
text_buffer = ""               # 入力された文章を格納するバッファ
last_prediction = None         # 直前に予測した文字
last_input_time = 0            # 最後に文字を追加した時間
stability_counter = 0          # 静止状態の継続フレームカウンター
is_stable = False              # 現在手が静止しているかどうかのフラグ
prev_raw_coords = None         # 前フレームの生座標

cap = cv2.VideoCapture(0)

print("\n==================================================")
print("リアルタイム手話ひらがな判別デモを起動しました。")
print("【操作方法】")
print(" - カメラに向かって指文字を作ります。")
print(" - 手の動きを『ピタッ』と止めると、文字が入力されます。")
print(" - [Space]キー: 入力テキストをクリア")
print(" - [Backspace]キー: 最後の1文字を消去")
print(" - [q]キー: プログラムを終了")
print("==================================================\n")

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("カメラから映像を取得できませんでした。")
            break

        # 鏡のように表示するため水平反転し、RGBに変換
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 現在のフレームの右手・左手の座標データ (合計126次元)
        right_hand_data = [0.0] * 63
        left_hand_data = [0.0] * 63
        
        has_hands = False

        if results.multi_hand_landmarks and results.multi_handedness:
            has_hands = True
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                label = handedness.classification[0].label
                temp_coords = []
                for landmark in hand_landmarks.landmark:
                    temp_coords.extend([landmark.x, landmark.y, landmark.z])
                
                if label == "Right":
                    right_hand_data = temp_coords
                elif label == "Left":
                    left_hand_data = temp_coords

        # 左右のデータを結合 (126次元の生座標)
        current_raw_coords = np.array(right_hand_data + left_hand_data)

        # --- 速度の計算（静止判定） ---
        velocity = 0.0
        if has_hands and prev_raw_coords is not None:
            # 前のフレームと現在のフレームの差分を計算
            diff = current_raw_coords - prev_raw_coords
            # 各関節(計42点)のユークリッド距離の平均を「速度」とする
            diff_reshaped = diff.reshape(42, 3)
            # すべて0(手が映っていない側)の関節は速度計算の平均から除外
            valid_joints = np.any(prev_raw_coords.reshape(42, 3) != 0.0, axis=1)
            if np.sum(valid_joints) > 0:
                distances = np.linalg.norm(diff_reshaped[valid_joints], axis=1)
                velocity = np.mean(distances)

        prev_raw_coords = current_raw_coords if has_hands else None

        # --- 静止・動作切り替えロジック ---
        current_time = time.time()
        prediction_text = "Tracking..."
        confidence_text = ""

        if has_hands and velocity > 0.0:
            if velocity < VELOCITY_THRESHOLD:
                stability_counter += 1
            else:
                stability_counter = 0
                is_stable = False  # 動かすと静止判定はリセット
            
            # 一定フレーム静止し続けた場合
            if stability_counter >= STABILITY_FRAMES:
                is_stable = True
        else:
            stability_counter = 0
            is_stable = False

        # --- モデル推論と文字入力トリガー ---
        if is_stable and model is not None:
            # 1. 座標を正規化 (手首原点、サイズスケーリング)
            r_norm = normalize_hand_data(np.array(right_hand_data))
            l_norm = normalize_hand_data(np.array(left_hand_data))
            normalized_features = np.concatenate([r_norm, l_norm]).reshape(1, -1)

            # 2. モデルによる予測
            probs = model.predict_proba(normalized_features)[0]
            max_idx = np.argmax(probs)
            pred_class = model.classes_[max_idx]
            confidence = probs[max_idx]

            prediction_text = f"Pred: {pred_class}"
            confidence_text = f"Conf: {confidence * 100:.1f}%"

            # 3. 入力トリガーの条件判定
            # 信頼度がしきい値を超えており、
            # (前回の入力と異なる文字、または前回と同じ文字ならクールダウン時間を経過している場合)
            if confidence >= CONFIDENCE_THRESHOLD:
                if pred_class != last_prediction or (current_time - last_input_time) > INPUT_COOLDOWN:
                    text_buffer += pred_class
                    last_prediction = pred_class
                    last_input_time = current_time
                    # 決定音の代わりにコンソール表示
                    print(f"▶ 入力されました: '{pred_class}' (現在値: {text_buffer})")
                    # 二重入力を防ぐため、一度 stability_counter を調整
                    stability_counter = -5  # 少しの間、入力をロックする

        elif model is None:
            prediction_text = "Model not trained"

        # --- 画面描画 (HUD / UI) ---
        # 1. テキストバッファ (入力された文章)
        cv2.rectangle(image, (0, 0), (640, 60), (30, 30, 30), -1)
        cv2.putText(image, f"Text: {text_buffer}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # 2. 状態インジケータ
        status_color = (0, 255, 0) if is_stable else (0, 165, 255)
        status_name = "STABLE (Input)" if is_stable else "MOVING"
        cv2.putText(image, f"Status: {status_name}", (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(image, f"Speed: {velocity:.5f} (Thresh: {VELOCITY_THRESHOLD})", (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 3. 予測結果と信頼度
        if has_hands:
            cv2.putText(image, prediction_text, (15, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(image, confidence_text, (15, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)

        # 画面表示
        cv2.imshow('Hand Sign Word Input Demo', image)

        # キー入力制御
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # スペースキーでテキストをクリア
            text_buffer = ""
            last_prediction = None
            print("▶ テキストをクリアしました")
        elif key == 8 or key == 127:  # Backspace キーで1文字消去 (環境によってキーコードが異なる)
            text_buffer = text_buffer[:-1]
            last_prediction = None
            print(f"▶ 1文字消去しました (現在値: {text_buffer})")

cap.release()
cv2.destroyAllWindows()
