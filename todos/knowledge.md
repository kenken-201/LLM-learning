# LLM学習 確度別ナレッジ＆技術メモ（knowledge）

> **運用ルール**:
> - 実装・学習実験で確認された確定事項や外部仕様（PyTorch仕様、CUDA制約、スループット測定値など）は `## [VERIFIED FACTS]` に記録する。
> - 検討中の仮説、実験結果への疑問、AIへのプロンプト回答の要約は `## [HYPOTHESES & OPEN QUESTIONS]` に記載する。

---

## [VERIFIED FACTS]（検証済み事実・確定した外部制約）

### 1. 開発・実行環境のベースライン
- **ローカル環境**: macOS（Apple Silicon / Staff）。CPU / MPS（Metal Performance Shaders）での動作確認が可能。
- **Phase 0 の制約**: micrograd 自作およびテンソル演算は CPU / numpy で完結可能（GPU不要）。
- **Phase 2 以降の計算資源**: 単一GPU（Google Colab Pro T4/L4、またはスポット A100 40GB）を時間単価（$1〜1.5/h）で必要時のみ利用する前提。

### 2. リポジトリ構成（SSOT）
- **教材正本**: [`docs/`](file:///Users/sasaki/Desktop/progDoc/LLM-learning/docs) 配下に Phase 0〜4、移行条件、付録A〜Cを無損失で配置済み。
- **進行管理**: [`todos/handover.md`](file:///Users/sasaki/Desktop/progDoc/LLM-learning/todos/handover.md) を起点としてセッション間を引き継ぐ。

---

## [HYPOTHESES & OPEN QUESTIONS]（未決事項・仮説・検討事項）

### 1. 環境構築ツールチェーンの選定
- **仮説**: Python パッケージ管理には `uv`（Astral製）を使用すると、仮想環境作成と依存関係解決が数秒で完了し、決定論的な `uv.lock` により環境再現性が最大化される。
- **次セッションでの確認**: `uv` の導入可否、または標準 `venv` + `pip` のどちらを本リポジトリの標準とするか。
