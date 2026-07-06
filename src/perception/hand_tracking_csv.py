import cv2
import mediapipe as mp
import csv
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- 【設定】記録したい手話の単語名 ---
WORD_NAME = "hello_both_hands"  # 両手用のテスト名
CSV_FILE_NAME = f"sign_language_{WORD_NAME}.csv"

cap = cv2.VideoCapture(0)

# CSVファイルのヘッダー（列名）を作成
# 右手(right_joint_0〜20) と 左手(left_joint_0〜20) のすべての座標を用意
with open(CSV_FILE_NAME, mode='w', newline='') as f:
    writer = csv.writer(f)
    header = ["timestamp"]
    # 右手の21箇所
    for i in range(21):
        header.extend([f"R_joint_{i}_x", f"R_joint_{i}_y", f"R_joint_{i}_z"])
    # 左手の21箇所
    for i in range(21):
        header.extend([f"L_joint_{i}_x", f"L_joint_{i}_y", f"L_joint_{i}_z"])
    writer.writerow(header)

print(f"両手データ収集モードを開始しました：単語【{WORD_NAME}】")
print("スペースキーを押している間だけデータを保存します。終了は 'q' です。")

is_recording = False

with mp_hands.Hands(
    max_num_hands=2,              # ★【変更】最大2本の手を検出
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 画面上の録画状態表示
        if is_recording:
            cv2.putText(image, "RECORDING...", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(image, "Press SPACE to Record", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 1フレームごとのデータを初期化（最初はすべて0を入れておく）
        # 手が画面に映っていないときは0が保存されるようにします
        right_hand_data = [0.0] * 63
        left_hand_data = [0.0] * 63

        if results.multi_hand_landmarks and results.multi_handedness:
            # 検出された手と、その左右の判定をループ処理
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # 骨格を描画
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # AIが判定した左右のラベルを取得（"Left" または "Right"）
                # ※カメラ反転の関係で、実際の右手・左手と一致させる処理
                label = handedness.classification[0].label
                
                # 21箇所の座標を1つのリストにまとめる
                temp_coords = []
                for landmark in hand_landmarks.landmark:
                    temp_coords.extend([landmark.x, landmark.y, landmark.z])
                
                # 左右に応じてデータを格納
                if label == "Right":
                    right_hand_data = temp_coords
                elif label == "Left":
                    left_hand_data = temp_coords

            # スペースキーが押されている場合のみCSVに保存
            if is_recording:
                # タイムスタンプ + 右手データ + 左手データ を合体
                row_data = [time.time()] + right_hand_data + left_hand_data
                with open(CSV_FILE_NAME, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row_data)

        cv2.imshow('Both Hands Data Collector', image)
        
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            is_recording = not is_recording
            if is_recording:
                print(">> 両手録画開始")
            else:
                print(">> 両手録画停止・保存完了")

cap.release()
cv2.destroyAllWindows()