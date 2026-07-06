import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class VLADataset(Dataset):
    """
    VLA（Vision-Language-Action）モデル用のカスタムデータセット。
    アプローチB（連続値）とアプローチC（離散トークン）の双方に対応。
    """

    def __init__(self, jsonl_path, approach="discrete", force_simple_prompt=False):
        self.records = []
        self.approach = approach
        self.force_simple_prompt = force_simple_prompt

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line.strip()))

        print(
            f"[{approach.upper()}] データセットをロードしました。件数: {len(self.records)}"
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        instruction = record["instruction"]

        if self.force_simple_prompt:
            instruction = f"ClassID_{idx % 10:02d}"

        if self.approach == "discrete":
            # アプローチC（離散トークン）: 文字列からポーズトークンIDのリストを取り出す
            output_str = record["output"]
            tokens = [
                int(tok.replace("<pose_", "").replace(">", ""))
                for tok in output_str.split(" ")
            ]

            # 最大長さにパディング
            max_len = 100
            padded_tokens = tokens[:max_len] + [0] * max(0, max_len - len(tokens))

            return {
                "instruction": instruction,
                "targets": torch.tensor(padded_tokens, dtype=torch.long),
            }
        else:
            # アプローチB（連続座標）: 126次元座標のシーケンス
            actions = record["actions"]
            max_len = 100
            padded_actions = actions[:max_len] + [[0.0] * 126] * max(
                0, max_len - len(actions)
            )

            return {
                "instruction": instruction,
                "targets": torch.tensor(padded_actions, dtype=torch.float32),
            }


class VLAModuleMock(nn.Module):
    """
    VLAモデルのAction層をシミュレートする簡易トランスフォーマー。
    """

    def __init__(self, vocab_size=64, embed_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True),
            num_layers=2,
        )
        self.action_head = nn.Linear(embed_dim, vocab_size)
        self.regress_head = nn.Linear(embed_dim, 126)

    def forward(self, x, mode="discrete"):
        # 回帰モードの入力はダミーインデックス
        if mode == "continuous":
            x = x.clamp(0, 63)
        x_embed = self.embedding(x)
        x_trans = self.transformer(x_embed)
        if mode == "discrete":
            return self.action_head(x_trans)
        else:
            return self.regress_head(x_trans)


def run_single_experiment(dataset_path, approach="discrete", rank=16, simple_prompt=False):
    """
    1セッションの訓練シミュレーションを実行し、Loss履歴を返却します。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = VLADataset(dataset_path, approach=approach, force_simple_prompt=simple_prompt)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = VLAModuleMock().to(device)
    # LoRAのランク設定変更による影響を学習率に反映させて擬似シミュレート
    lr = 1e-3 * (16 / rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    if approach == "discrete":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    model.train()
    epoch_list = []
    loss_history = []

    # シミュレーション訓練（3エポック）
    for epoch in range(3):
        epoch_loss = 0.0
        for batch in dataloader:
            targets = batch["targets"].to(device)
            optimizer.zero_grad()

            if approach == "discrete":
                dummy_input = torch.zeros_like(targets).to(device)
                outputs = model(dummy_input, mode="discrete")
                loss = criterion(outputs.view(-1, 64), targets.view(-1))
            else:
                dummy_input = torch.zeros(
                    (targets.shape[0], targets.shape[1]), dtype=torch.long
                ).to(device)
                outputs = model(dummy_input, mode="continuous")
                loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, len(dataloader))
        epoch_list.append(epoch + 1)
        loss_history.append(avg_loss)
        print(f"  Epoch {epoch+1:02d}/03 - Simulated Loss: {avg_loss:.4f}")

    return epoch_list, loss_history


def plot_comparison_results(results, experiment_type):
    """
    2つのアプローチの比較学習曲線を同一のグラフ上に重ねて出力・保存します。
    """
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    colors = ['#1e3a8a', '#10b981']
    markers = ['o', 's']

    for i, (label, data) in enumerate(results.items()):
        ax.plot(
            data["epochs"],
            data["losses"],
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            linewidth=2.5,
            markersize=7,
            label=label
        )

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylabel('Simulated Training Loss', fontsize=12, fontweight='bold', labelpad=8)
    ax.grid(True, linestyle=':', alpha=0.6, color='#cbd5e1')
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#e2e8f0', fontsize=11)

    graph_title = f"VLA Comparative Study: {experiment_type.replace('_', ' ').title()}"
    plt.title(graph_title, fontsize=13, fontweight='bold', pad=12)

    fig.tight_layout()
    output_name = f"vla_comparison_{experiment_type}.png"
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"\n📊 比較評価グラフを保存しました: {output_name}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compare",
        type=str,
        default="compare_data_format",
        choices=["compare_data_format", "compare_lora_rank", "compare_prompt_style"],
        help="比較対象実験タイプ"
    )
    args = parser.parse_args()

    results = {}

    if args.compare == "compare_data_format":
        print("\n--- 🔄 実験A: データ表現比較のローカルシミュレーション (Bins vs Poses) ---")
        conditions = {
            "Continuous Action Bins (Approach B)": {"dataset": "data/processed/vla_continuous.jsonl", "approach": "continuous", "rank": 16, "simple_prompt": False},
            "Discrete Pose Tokens (Approach C)": {"dataset": "data/processed/vla_discrete.jsonl", "approach": "discrete", "rank": 16, "simple_prompt": False}
        }
    elif args.compare == "compare_lora_rank":
        print("\n--- 🔄 実験B: Lの適合容量評価のローカルシミュレーション (LoRA Rank 8 vs 32) ---")
        conditions = {
            "LoRA Rank = 8": {"dataset": "data/processed/vla_discrete.jsonl", "approach": "discrete", "rank": 8, "simple_prompt": False},
            "LoRA Rank = 32": {"dataset": "data/processed/vla_discrete.jsonl", "approach": "discrete", "rank": 32, "simple_prompt": False}
        }
    else:
        print("\n--- 🔄 実験C: プロンプト記述セマンティクスのローカルシミュレーション ---")
        conditions = {
            "Simple ID Prompt (No Semantic)": {"dataset": "data/processed/vla_discrete.jsonl", "approach": "discrete", "rank": 16, "simple_prompt": True},
            "Natural Japanese Instruction": {"dataset": "data/processed/vla_discrete.jsonl", "approach": "discrete", "rank": 16, "simple_prompt": False}
        }

    for label, cond in conditions.items():
        print(f"\n🚀 実験条件: {label}")
        epochs, losses = run_single_experiment(
            cond["dataset"],
            approach=cond["approach"],
            rank=cond["rank"],
            simple_prompt=cond["simple_prompt"]
        )
        results[label] = {"epochs": epochs, "losses": losses}

    plot_comparison_results(results, args.compare)


if __name__ == "__main__":
    main()
