"""両手42点の座標データを立体的な手の形状（アバター）として描画・再生する3Dビューア。

主な機能:
- 手のひらメッシュ（ポリゴンサーフェス）と肉付けされた指（立体アバター）による直感的な手の可視化
- タイムスタンプに基づく線形補間アニメーション再生
- Spaceキーで再生/停止、←/→でコマ送り、スライダーで任意フレームへのシーク
- CSV / JSONL / NPY / トークン列の読み込み対応

起動方法:
    python src/viewer/hand_viewer.py                       # ファイル選択ダイアログ
    python src/viewer/hand_viewer.py data/raw/hiragana/sign_language_a_1.csv
    python src/viewer/hand_viewer.py data/processed/vla_discrete_64.jsonl --index 0
"""
import argparse
import os
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from motion_data import load_motion, motion_from_tokens

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "raw")

# MediaPipe Hands の各指の関節インデックス
FINGERS = [
    [0, 1, 2, 3, 4],       # 親指
    [0, 5, 6, 7, 8],       # 人差し指
    [0, 9, 10, 11, 12],    # 中指
    [0, 13, 14, 15, 16],   # 薬指
    [0, 17, 18, 19, 20],   # 小指
]

# 手のひら・手の甲を構成するポリゴン面（頂点インデックス）
PALM_POLYGONS = [
    [0, 1, 2, 5],      # 手首 - 親指基部 - 人差し指基部
    [0, 5, 9],         # 手首 - 人差し指基部 - 中指基部
    [0, 9, 13],        # 手首 - 中指基部 - 薬指基部
    [0, 13, 17],       # 手首 - 薬指基部 - 小指基部
    [5, 9, 13, 17],    # 指の付け根アーチ（手のひら上部）
]

HAND_CONFIG = {
    "R": {
        "slice": slice(0, 21),
        "flesh_color": "#38bdf8",   # 肉付け（スカイブルー）
        "bone_color": "#0369a1",    # 骨格芯（ディープブルー）
        "palm_color": "#7dd3fc",    # 手のひら面
        "joint_color": "#e0f2fe",   # 関節球
        "tip_color": "#0284c7",     # 指先
        "label": "右手"
    },
    "L": {
        "slice": slice(21, 42),
        "flesh_color": "#fb923c",   # 肉付け（オレンジ）
        "bone_color": "#c2410c",    # 骨格芯（ダークオレンジ）
        "palm_color": "#fdba74",    # 手のひら面
        "joint_color": "#ffedd5",   # 関節球
        "tip_color": "#ea580c",     # 指先
        "label": "左手"
    }
}

SPEEDS = [0.25, 0.5, 1.0, 2.0]
TICK_MS = 30
TRAIL_SEC = 1.0
NORMALIZED_OFFSET = 1.4
DEFAULT_VIEW = (15, -70)

for key in ("keymap.back", "keymap.forward", "keymap.home", "keymap.zoom"):
    plt.rcParams[key] = []


def use_japanese_font():
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Hiragino Sans", "Hiragino Kaku Gothic ProN", "YuGothic", "Osaka", "Yu Gothic", "Meiryo", "Noto Sans CJK JP", "IPAexGothic"):
        if name in available:
            plt.rcParams["font.family"] = name
            return


use_japanese_font()


def ask_file():
    if sys.platform == "darwin":
        script = f'POSIX path of (choose file with prompt "座標データ (CSV / JSONL / NPY) を選択" default location (POSIX file "{RAW_DIR}"))'
        result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
        return result.stdout.strip() or None

    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="座標データを選択", initialdir=RAW_DIR,
        filetypes=[("座標データ", "*.csv *.jsonl *.npy"), ("すべて", "*.*")],
    )
    root.destroy()
    return path or None


class HandViewer:
    def __init__(self, motion, path=None, fps=None, quantizer=None):
        self.path, self.fps, self.quantizer = path, fps, quantizer
        self.playing = False
        self.speed_idx = SPEEDS.index(1.0)
        self.show_trail = False
        self.t = 0.0
        self.last_tick = time.perf_counter()

        self.fig = plt.figure(figsize=(9, 8), facecolor="#0f172a")
        self.fig.canvas.manager.set_window_title("3D Hand Avatar Viewer")
        self.ax = self.fig.add_axes([0.0, 0.16, 1.0, 0.76], projection="3d")
        self.ax.set_facecolor("#0f172a")

        self.artists = {}
        for side, cfg in HAND_CONFIG.items():
            # 手のひらメッシュ
            palm_poly = Poly3DCollection([], alpha=0.55, facecolor=cfg["palm_color"], edgecolor=cfg["bone_color"], linewidths=1.0)
            self.ax.add_collection3d(palm_poly)

            # 指の肉付け（太いライン）
            flesh_lines = [self.ax.plot([], [], [], color=cfg["flesh_color"], lw=8.5, solid_capstyle="round",
                                       alpha=0.85, label=cfg["label"] if i == 0 else None)[0]
                           for i in range(len(FINGERS))]

            # 指の骨格芯（内部の線）
            bone_lines = [self.ax.plot([], [], [], color=cfg["bone_color"], lw=2.0, alpha=0.9)[0]
                          for i in range(len(FINGERS))]

            # 関節球
            joints = self.ax.plot([], [], [], "o", color=cfg["joint_color"], markeredgecolor=cfg["bone_color"],
                                  markeredgewidth=1.0, ms=6.5)[0]

            # 指先のハイライト
            tips = self.ax.plot([], [], [], "o", color=cfg["tip_color"], markeredgecolor="#ffffff",
                                markeredgewidth=1.2, ms=9.0)[0]

            # 移動軌跡
            trail = self.ax.plot([], [], [], color=cfg["flesh_color"], lw=1.5, alpha=0.5)[0]

            self.artists[side] = {
                "palm": palm_poly,
                "flesh": flesh_lines,
                "bones": bone_lines,
                "joints": joints,
                "tips": tips,
                "trail": trail,
            }

        legend = self.ax.legend(loc="upper right", facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
        if legend:
            for text in legend.get_texts():
                text.set_color("#e2e8f0")

        self.title = self.fig.text(0.03, 0.965, "", fontsize=11, fontweight="bold", va="top", color="#f8fafc")
        self.status = self.fig.text(0.03, 0.932, "", fontsize=9, va="top", color="#94a3b8")
        self.fig.text(0.03, 0.015, "Space: 再生/停止 | ←→: コマ送り | ↑↓: 速度 | [ ]: レコード切替 | "
                      "O: 開く | R: 視点リセット | T: 軌跡", fontsize=8, color="#64748b")

        slider_ax = self.fig.add_axes([0.1, 0.105, 0.8, 0.03], facecolor="#1e293b")
        self.slider = Slider(slider_ax, "フレーム", 0, 1, valinit=0, valstep=1, valfmt="%d", color="#38bdf8")
        self.slider.label.set_color("#cbd5e1")
        self.slider.valtext.set_color("#f8fafc")
        self.slider.on_changed(self.on_seek)

        specs = [("開く", self.open_file), ("◀", lambda: self.step(-1)), ("再生", self.toggle_play),
                 ("▶", lambda: self.step(1)), ("1.0x", self.cycle_speed), ("視点リセット", self.reset_view),
                 ("軌跡", self.toggle_trail)]
        self.buttons = {}
        width, gap, left = 0.115, 0.012, 0.06
        for i, (label, func) in enumerate(specs):
            btn_ax = self.fig.add_axes([left + i * (width + gap), 0.04, width, 0.045])
            btn = Button(btn_ax, label, color="#1e293b", hovercolor="#334155")
            btn.label.set_color("#f8fafc")
            btn.label.set_fontsize(8.5)
            btn.on_clicked(lambda _event, f=func: f())
            self.buttons[label] = btn
        self.play_label = self.buttons["再生"].label
        self.speed_label = self.buttons["1.0x"].label

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.timer = self.fig.canvas.new_timer(interval=TICK_MS)
        self.timer.add_callback(self.tick)
        self.timer.start()

        self.reset_view()
        self.set_motion(motion)

    # ---- データ設定 ----
    def set_motion(self, motion):
        self.motion = motion
        n = len(motion.frames)
        self.slider.valmax = max(n - 1, 1)
        self.slider.ax.set_xlim(0, self.slider.valmax)

        # 最初のフレームが未検出(全0)の場合、手が最初に現れるフレームに初期シークする
        first_valid_idx = 0
        for idx, f in enumerate(motion.frames):
            if f.any():
                first_valid_idx = idx
                break

        self.t = float(motion.times[first_valid_idx])
        self.fit_axes()
        self.title.set_text(motion.title)
        self.redraw()

    def reload(self, path, index=0):
        try:
            motion = load_motion(path, index, self.fps, self.quantizer)
        except Exception as e:
            self.status.set_text(f"読み込み失敗: {e}")
            self.fig.canvas.draw_idle()
            return
        self.path = path
        self.set_motion(motion)

    def to_display(self, pts, side):
        pts = np.array(pts, dtype=float)
        if self.motion.normalized:
            pts[..., 0] += NORMALIZED_OFFSET if side == "R" else -NORMALIZED_OFFSET
        # MediaPipe は x:右, y:下, z:奥行き。描画では上方向を Z 軸にする
        return pts[..., 0], pts[..., 2], -pts[..., 1]

    def fit_axes(self):
        chunks = []
        for side, cfg in HAND_CONFIG.items():
            hand = self.motion.frames[:, cfg["slice"]]
            hand = hand[hand.any(axis=(1, 2))]
            if len(hand):
                chunks.append(np.column_stack(self.to_display(hand.reshape(-1, 3), side)))
        if not chunks:
            # 手が存在しない場合のフォールバック境界
            self.ax.set_xlim(-0.5, 0.5)
            self.ax.set_ylim(-0.5, 0.5)
            self.ax.set_zlim(-0.5, 0.5)
            return

        pts = np.concatenate(chunks)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        center = (lo + hi) / 2
        half = max((hi - lo).max() / 2 * 1.15, 0.15)
        self.ax.set_xlim(center[0] - half, center[0] + half)
        self.ax.set_ylim(center[1] - half, center[1] + half)
        self.ax.set_zlim(center[2] - half, center[2] + half)
        self.ax.set_box_aspect((1, 1, 1))

        # 軸の見た目
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor("#334155")
        self.ax.yaxis.pane.set_edgecolor("#334155")
        self.ax.zaxis.pane.set_edgecolor("#334155")
        self.ax.tick_params(colors="#64748b", labelsize=7)
        self.ax.set_xlabel("X (左右)", color="#94a3b8", fontsize=8)
        self.ax.set_ylabel("Z (奥行き)", color="#94a3b8", fontsize=8)
        self.ax.set_zlabel("Y (上下)", color="#94a3b8", fontsize=8)

    # ---- 描画 ----
    def frame_at(self, t):
        times = self.motion.times
        if len(times) == 1:
            return 0, 0.0
        i = int(np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 2))
        frac = float(np.clip((t - times[i]) / (times[i + 1] - times[i]), 0.0, 1.0))
        if frac >= 1.0:
            return i + 1, 0.0
        return i, frac

    def redraw(self, sync_slider=True):
        frames, times = self.motion.frames, self.motion.times
        i, frac = self.frame_at(self.t)
        j = min(i + 1, len(frames) - 1)
        detected = []

        for side, cfg in HAND_CONFIG.items():
            arts = self.artists[side]
            sl = cfg["slice"]
            a, b = frames[i, sl], frames[j, sl]

            if not a.any():
                pts = None
            elif b.any():
                pts = a + (b - a) * frac
            else:
                pts = a

            if pts is None:
                # 非検出フレーム
                arts["palm"].set_verts([])
                for line in arts["flesh"] + arts["bones"]:
                    line.set_data_3d([], [], [])
                arts["joints"].set_data_3d([], [], [])
                arts["tips"].set_data_3d([], [], [])
            else:
                detected.append(cfg["label"])
                x, y, z = self.to_display(pts, side)
                points_3d = np.column_stack([x, y, z])

                # 1. 手のひらメッシュの更新
                verts = [[points_3d[poly_idx] for poly_idx in poly] for poly in PALM_POLYGONS]
                arts["palm"].set_verts(verts)

                # 2. 指の肉付けライン & 骨格芯ラインの更新
                for f_line, b_line, f_idx in zip(arts["flesh"], arts["bones"], FINGERS):
                    f_line.set_data_3d(x[f_idx], y[f_idx], z[f_idx])
                    b_line.set_data_3d(x[f_idx], y[f_idx], z[f_idx])

                # 3. 関節球
                arts["joints"].set_data_3d(x, y, z)

                # 4. 指先ハイライト (4: 親指, 8: 人差し指, 12: 中指, 16: 薬指, 20: 小指)
                tip_idx = [4, 8, 12, 16, 20]
                arts["tips"].set_data_3d(x[tip_idx], y[tip_idx], z[tip_idx])

            # 人差し指軌跡
            tips_hist = frames[np.searchsorted(times, self.t - TRAIL_SEC):i + 1, sl][:, 8]
            tips_hist = tips_hist[tips_hist.any(axis=1)]
            if self.show_trail and len(tips_hist) > 1:
                arts["trail"].set_data_3d(*self.to_display(tips_hist, side))
            else:
                arts["trail"].set_data_3d([], [], [])

        state = "再生中" if self.playing else "停止中"
        hands_str = " / ".join(detected) if detected else "手未検出（カメラ準備中または非検出フレーム）"
        self.status.set_text(
            f"フレーム: {i + 1}/{len(frames)}  |  時間: {self.t:.2f}s/{times[-1]:.2f}s  |  "
            f"速度: {SPEEDS[self.speed_idx]}x  [{state}]  |  {hands_str}"
        )
        if sync_slider:
            self.slider.eventson = False
            self.slider.set_val(i)
            self.slider.eventson = True
        self.fig.canvas.draw_idle()

    # ---- 再生制御 ----
    def tick(self):
        now = time.perf_counter()
        dt, self.last_tick = now - self.last_tick, now
        if not self.playing:
            return
        self.t += dt * SPEEDS[self.speed_idx]
        if self.t > self.motion.times[-1]:
            self.t = 0.0
        self.redraw()

    def toggle_play(self):
        self.playing = not self.playing
        if self.playing and self.t >= self.motion.times[-1]:
            self.t = 0.0
        self.last_tick = time.perf_counter()
        self.play_label.set_text("一時停止" if self.playing else "再生")
        self.redraw()

    def step(self, delta):
        self.playing = False
        self.play_label.set_text("再生")
        i, frac = self.frame_at(self.t)
        if delta < 0 and frac > 0:
            delta = 0
        i = int(np.clip(i + delta, 0, len(self.motion.times) - 1))
        self.t = float(self.motion.times[i])
        self.redraw()

    def seek_end(self, end):
        self.t = float(self.motion.times[-1] if end else 0.0)
        self.redraw()

    def on_seek(self, val):
        self.t = float(self.motion.times[int(val)])
        self.redraw(sync_slider=False)

    def cycle_speed(self, delta=1):
        self.speed_idx = (self.speed_idx + delta) % len(SPEEDS)
        self.speed_label.set_text(f"{SPEEDS[self.speed_idx]}x")
        self.redraw()

    def toggle_trail(self):
        self.show_trail = not self.show_trail
        self.redraw()

    def reset_view(self):
        self.ax.view_init(*DEFAULT_VIEW)
        self.fig.canvas.draw_idle()

    def open_file(self):
        self.playing = False
        self.play_label.set_text("再生")
        path = ask_file()
        if path:
            self.reload(path)
        else:
            self.redraw()

    def change_record(self, delta):
        if self.path and self.motion.n_records > 1:
            self.reload(self.path, self.motion.index + delta)

    def on_key(self, event):
        actions = {
            " ": self.toggle_play,
            "left": lambda: self.step(-1),
            "right": lambda: self.step(1),
            "up": lambda: self.cycle_speed(1),
            "down": lambda: self.cycle_speed(-1),
            "home": lambda: self.seek_end(False),
            "end": lambda: self.seek_end(True),
            "[": lambda: self.change_record(-1),
            "]": lambda: self.change_record(1),
            "o": self.open_file,
            "r": self.reset_view,
            "t": self.toggle_trail,
        }
        if event.key in actions:
            actions[event.key]()


def main():
    parser = argparse.ArgumentParser(description="手話の3D手骨格アバター再生ビューア")
    parser.add_argument("path", nargs="?", help="CSV / JSONL / NPY ファイル。省略するとダイアログで選択")
    parser.add_argument("--index", type=int, default=0, help="JSONL のレコード番号 (0始まり)")
    parser.add_argument("--fps", type=float, default=None, help="固定フレームレートで再生")
    parser.add_argument("--quantizer", default=None, help="<pose_N> トークンの復元に使う量子化器 (.joblib)")
    parser.add_argument("--tokens", default=None, help='モデル出力のトークン列 例: "<pose_3> <pose_12>"')
    args = parser.parse_args()

    use_japanese_font()
    path = None
    try:
        if args.tokens:
            motion = motion_from_tokens(args.tokens, args.quantizer, args.fps)
        else:
            path = args.path or ask_file()
            if not path:
                print("ファイルが選択されませんでした")
                return
            motion = load_motion(path, args.index, args.fps, args.quantizer)
    except Exception as e:
        sys.exit(f"読み込み失敗: {e}")

    viewer = HandViewer(motion, path, args.fps, args.quantizer)  # noqa: F841
    plt.show()


if __name__ == "__main__":
    main()
