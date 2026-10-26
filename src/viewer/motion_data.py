"""手の座標データを読み込み、ビューアで扱う (N, 42, 3) 配列にそろえる。

対応形式
- 収録CSV (data/raw/**/sign_language_*.csv)
- VLA用JSONL: actions(連続値) または output の <pose_N> トークン列
- NumPy配列 .npy: モデルの予測結果などを (N, 126) で保存したもの

関節の並びは 0-20 が右手、21-41 が左手 (MediaPipe Hands の番号順)。
"""
import json
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_FPS = 10.0
COORD_COLS = [f"{side}_joint_{i}_{axis}" for side in "RL" for i in range(21) for axis in "xyz"]
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "learning", "weights")


@dataclass
class Motion:
    times: np.ndarray   # (N,) 先頭フレームを0とした秒数
    frames: np.ndarray  # (N, 42, 3)
    title: str = ""
    normalized: bool = False  # 手首原点の正規化済みデータかどうか
    index: int = 0            # JSONL内のレコード番号
    n_records: int = 1


def fixed_rate_times(n, fps):
    return np.arange(n) / float(fps)


def is_normalized(frames):
    hands = frames.reshape(-1, 21, 3)
    present = hands.any(axis=(1, 2))
    return bool(present.any() and np.allclose(hands[present, 0], 0.0, atol=1e-6))


def actions_to_frames(actions):
    arr = np.asarray(actions, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 126:
        raise ValueError(f"126次元のフレーム列を想定していますが、形状が {arr.shape} でした")
    return arr.reshape(-1, 42, 3)


def parse_tokens(text):
    return [int(t) for t in re.findall(r"<pose_(\d+)>", text)]


def default_quantizer_path(jsonl_path=None):
    k = "64"
    if jsonl_path:
        m = re.search(r"_(\d+)\.jsonl$", os.path.basename(jsonl_path))
        if m:
            k = m.group(1)
    return os.path.join(WEIGHTS_DIR, f"pose_quantizer_{k}.joblib")


def tokens_to_frames(tokens, quantizer_path):
    import joblib

    if not os.path.exists(quantizer_path):
        raise FileNotFoundError(f"量子化器が見つかりません: {quantizer_path}")
    centers = np.asarray(joblib.load(quantizer_path)["cluster_centers_"])
    if centers.shape[1] != 126:
        raise ValueError(f"量子化器の次元が {centers.shape[1]} です (126次元の座標で学習したものが必要)")
    tokens = np.asarray(tokens, dtype=int)
    if tokens.size and tokens.max() >= len(centers):
        raise ValueError(f"トークンID {tokens.max()} が量子化器のクラスタ数 {len(centers)} を超えています")
    return actions_to_frames(centers[tokens])


def load_csv(path, fps=None):
    df = pd.read_csv(path)
    missing = [c for c in COORD_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"座標列が足りません (例: {missing[0]})。両手42点の列を持つCSVを指定してください")
    frames = df[COORD_COLS].to_numpy(dtype=float).reshape(-1, 42, 3)

    times = None
    if fps is None and "timestamp" in df.columns:
        ts = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
        if not np.isnan(ts).any() and len(ts) > 1 and np.all(np.diff(ts) > 0):
            times = ts - ts[0]
    if times is None:
        times = fixed_rate_times(len(frames), fps or DEFAULT_FPS)

    return Motion(times, frames, os.path.basename(path), is_normalized(frames))


def load_jsonl(path, index=0, fps=None, quantizer_path=None):
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        raise ValueError("JSONLにレコードがありません")
    index %= len(records)
    rec = records[index]

    if "actions" in rec:
        frames = actions_to_frames(rec["actions"])
    else:
        tokens = parse_tokens(rec.get("output", ""))
        if not tokens:
            raise ValueError("actions も <pose_N> トークンも見つかりません")
        frames = tokens_to_frames(tokens, quantizer_path or default_quantizer_path(path))

    title = f"{os.path.basename(path)}  [{index + 1}/{len(records)}]  {rec.get('instruction', '')}"
    times = fixed_rate_times(len(frames), fps or DEFAULT_FPS)
    return Motion(times, frames, title, True, index, len(records))


def load_npy(path, fps=None):
    frames = actions_to_frames(np.load(path).reshape(-1, 126))
    times = fixed_rate_times(len(frames), fps or DEFAULT_FPS)
    return Motion(times, frames, os.path.basename(path), is_normalized(frames))


def motion_from_tokens(text, quantizer_path=None, fps=None):
    """モデルが出力した "<pose_3> <pose_3> <pose_12> ..." をそのまま再生用データにする。"""
    tokens = parse_tokens(text)
    if not tokens:
        raise ValueError("<pose_N> 形式のトークンが含まれていません")
    frames = tokens_to_frames(tokens, quantizer_path or default_quantizer_path())
    times = fixed_rate_times(len(frames), fps or DEFAULT_FPS)
    return Motion(times, frames, f"トークン列 ({len(tokens)} フレーム)", True)


def load_motion(path, index=0, fps=None, quantizer_path=None):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return load_csv(path, fps)
    if ext == ".jsonl":
        return load_jsonl(path, index, fps, quantizer_path)
    if ext == ".npy":
        return load_npy(path, fps)
    raise ValueError(f"対応していない拡張子です: {ext}")
