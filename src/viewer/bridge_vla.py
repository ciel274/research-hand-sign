"""VLAモデルの出力層とCGビューア・ロボットハンド(uHand)のインターフェースブリッジ。

主な機能:
1. VLA出力の解釈:
   - 離散トークン列 ("<pose_12> <pose_5> ...") -> 126次元座標系列への逆量子化
   - 連続アクションベクトル ((N, 126)) -> 関節座標への整形
2. ロボットハンド (uHand) 向け制御値の計算:
   - 手の3D骨格から各指（親指・人差し指・中指・薬指・小指）の屈曲角を算出 (0度: 伸展 〜 90度: 屈曲)
   - uHandシリアルコマンド形式 (PWM値 500〜2500μs, ID 1〜5/6) へのエンコード
3. CGビューアおよび他システムとの連携:
   - 変換したモーションを HandViewer で即時プレビュー
   - CSV / JSON 形式での出力
"""
import argparse
import json
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from motion_data import actions_to_frames, default_quantizer_path, tokens_to_frames, parse_tokens, Motion

# uHandの標準的なサーボID割り当て (1:親指, 2:人差し指, 3:中指, 4:薬指, 5:小指, 6:手首回転)
UHAND_SERVO_MAP = {
    "thumb": 1,
    "index": 2,
    "middle": 3,
    "ring": 4,
    "pinky": 5,
    "wrist": 6,
}


def compute_finger_flexion_angles(hand_coords_21x3):
    """21点の手の関節座標 (21, 3) から各指の屈曲角度 (度) を算出する。
    
    戻り値:
        dict: 各指の名前と屈曲角 (0: まっすぐ伸びている, 90: 直角に曲がっている, 最大約140度)
    """
    coords = np.asarray(hand_coords_21x3, dtype=float)
    if not coords.any():
        return {k: 0.0 for k in ("thumb", "index", "middle", "ring", "pinky")}

    def _angle(p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        # 180度(直線)なら屈曲0度、角度が狭まるほど曲がっているとみなす
        return float(np.degrees(np.pi - np.arccos(cos)))

    # 各指の代表的な関節(MCP, PIP, DIP)の角度
    # 親指: (1, 2, 3), 人差し指: (5, 6, 7), 中指: (9, 10, 11), 薬指: (13, 14, 15), 小指: (17, 18, 19)
    angles = {
        "thumb": _angle(coords[1], coords[2], coords[3]) + _angle(coords[2], coords[3], coords[4]) * 0.5,
        "index": _angle(coords[5], coords[6], coords[7]) + _angle(coords[6], coords[7], coords[8]) * 0.5,
        "middle": _angle(coords[9], coords[10], coords[11]) + _angle(coords[10], coords[11], coords[12]) * 0.5,
        "ring": _angle(coords[13], coords[14], coords[15]) + _angle(coords[14], coords[15], coords[16]) * 0.5,
        "pinky": _angle(coords[17], coords[18], coords[19]) + _angle(coords[18], coords[19], coords[20]) * 0.5,
    }
    # 0〜180の範囲にクリップ
    return {k: float(np.clip(v, 0.0, 180.0)) for k, v in angles.items()}


def flexion_to_uhand_pwm(flexion_deg, min_pwm=1500, max_pwm=2500, max_flexion=120.0):
    """指の屈曲角度 (0〜max_flexion度) を uHandサーボのPWMパルス幅 (μs) に変換する。
    
    通常 uHand は 1500μs付近が開き、2500μs(または500μs)付近が握りとなる。
    """
    ratio = np.clip(flexion_deg / max_flexion, 0.0, 1.0)
    pwm = min_pwm + ratio * (max_pwm - min_pwm)
    return int(np.clip(round(pwm), 500, 2500))


def generate_uhand_command(finger_angles, time_ms=100):
    """各指の角度辞書から uHand のシリアル制御プロトコル文字列を生成する。
    
    例: '#001P1500#002P1800#003P2100#004P1500#005P1500T0100!'
    """
    parts = []
    for finger, servo_id in UHAND_SERVO_MAP.items():
        if finger == "wrist":
            continue
        deg = finger_angles.get(finger, 0.0)
        pwm = flexion_to_uhand_pwm(deg)
        parts.append(f"#{servo_id:03d}P{pwm:04d}")
    parts.append(f"T{time_ms:04d}!")
    return "".join(parts)


def vla_output_to_frames(output_data, quantizer_path=None):
    """VLAの出力 (文字列トークン、トークンリスト、または126次元連続配列) を (N, 42, 3) 配列に変換する。"""
    if isinstance(output_data, str):
        tokens = parse_tokens(output_data)
        if not tokens:
            raise ValueError(f"有効なポーズトークンが見つかりません: {output_data[:100]}")
        q_path = quantizer_path or default_quantizer_path()
        return tokens_to_frames(tokens, q_path)
    elif isinstance(output_data, (list, tuple)):
        first = output_data[0]
        if isinstance(first, int):
            q_path = quantizer_path or default_quantizer_path()
            return tokens_to_frames(output_data, q_path)
        else:
            return actions_to_frames(output_data)
    elif isinstance(output_data, np.ndarray):
        return actions_to_frames(output_data)
    else:
        raise TypeError(f"未対応のデータ型です: {type(output_data)}")


def export_uhand_sequence(frames_nx42x3, side="R", time_step_ms=100):
    """(N, 42, 3) の時系列フレームから uHand コマンドのリストと角度ログを生成する。"""
    commands = []
    angles_log = []
    sl = slice(0, 21) if side == "R" else slice(21, 42)

    for i in range(len(frames_nx42x3)):
        hand = frames_nx42x3[i, sl]
        angles = compute_finger_flexion_angles(hand)
        cmd = generate_uhand_command(angles, time_ms=time_step_ms)
        commands.append(cmd)
        angles_log.append(angles)

    return commands, angles_log


def preview_in_viewer(frames, title="VLA Generated Motion", fps=10.0):
    """生成したモーションを即座に HandViewer で起動・確認する。"""
    from hand_viewer import HandViewer
    import matplotlib.pyplot as plt

    times = np.arange(len(frames)) / float(fps)
    motion = Motion(times=times, frames=frames, title=title, normalized=True)
    viewer = HandViewer(motion, fps=fps)  # noqa: F841
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="VLA出力とロボットハンド/3D描画の中継ブリッジ")
    parser.add_argument("--tokens", type=str, help='VLA出力のトークン列 例: "<pose_12> <pose_12> <pose_5>"')
    parser.add_argument("--jsonl", type=str, help="VLAのJSONLファイルパス")
    parser.add_argument("--index", type=int, default=0, help="JSONLのレコード番号")
    parser.add_argument("--preview", action="store_true", help="3Dビューアで再生する")
    parser.add_argument("--uhand", action="store_true", help="uHand用コマンドを出力する")
    parser.add_argument("--side", choices=["R", "L"], default="R", help="対象の手 (R または L)")
    args = parser.parse_args()

    frames = None
    title = "VLA Output"

    if args.tokens:
        frames = vla_output_to_frames(args.tokens)
        title = f"Tokens: {args.tokens[:40]}..."
    elif args.jsonl:
        with open(args.jsonl, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        rec = records[args.index % len(records)]
        title = f"{os.path.basename(args.jsonl)} [{args.index}]: {rec.get('instruction', '')}"
        if "output" in rec:
            frames = vla_output_to_frames(rec["output"])
        elif "actions" in rec:
            frames = vla_output_to_frames(rec["actions"])
    else:
        print("テスト用デフォルトトークン列を変換します...")
        dummy_tokens = "<pose_62> <pose_62> <pose_12> <pose_12> <pose_12>"
        frames = vla_output_to_frames(dummy_tokens)
        title = f"Sample: {dummy_tokens}"

    print(f"変換完了: フレーム数 = {len(frames)}, 骨格形状 = {frames.shape}")

    if args.uhand:
        cmds, angles = export_uhand_sequence(frames, side=args.side)
        print(f"\n--- uHand 制御コマンド (最初の3フレーム / 計 {len(cmds)} フレーム) ---")
        for i, (cmd, ang) in enumerate(zip(cmds[:3], angles[:3])):
            ang_str = ", ".join(f"{k}: {v:.1f}°" for k, v in ang.items())
            print(f"[{i}] 角度({ang_str}) -> コマンド: {cmd}")

    if args.preview:
        preview_in_viewer(frames, title=title)


if __name__ == "__main__":
    main()
