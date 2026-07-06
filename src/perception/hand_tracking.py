import cv2
import mediapipe as mp

# MediaPipeの手検出モジュールを初期化
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Macの内蔵カメラ（FaceTimeカメラ）からの入力を開始
cap = cv2.VideoCapture(0)

# MediaPipe Handsのコンテキストを開く
with mp_hands.Hands(
    max_num_hands=1,              # まずは片手でテスト
    min_detection_confidence=0.7,   # 検出のしきい値
    min_tracking_confidence=0.7    # 追跡のしきい値
) as hands:

    print("カメラを起動しました。終了するにはキーボードの 'q' を押してください。")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("カメラからの映像を取得できませんでした。")
            break

        # 画像を水平反転（鏡のように見せる）し、BGRからRGBに変換
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        
        # パフォーマンス向上のため一時的に書き込み不可にする
        image.flags.writeable = False
        
        # AIによる手骨格の検出を実行
        results = hands.process(image)

        # 描画のために書き込み可能に戻し、RGBからBGRに変換
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 手が検出された場合の処理
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 画面に手の骨格（ワイヤーフレーム）を描画
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                print("--- 新しいフレームの手の座標 ---")
                
                # 手首から指先まで21箇所の座標をループ処理
                for i, landmark in enumerate(hand_landmarks.landmark):
                    x = landmark.x
                    y = landmark.y
                    z = landmark.z # 画面からの奥行き
                    
                    # 人差し指の先端（ID: 8）のデータだけをコンソールに表示
                    if i == 8:
                        print(f"人差し指の先 (ID: {i}) -> X: {x:.4f}, Y: {y:.4f}, Z: {z:.4f}")

        # 映像をウィンドウに表示
        cv2.imshow('MediaPipe Hand Tracking', image)

        # 'q' キーが押されたらループを抜ける
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()