# LLM開発実践ロードマップ 進行管理表（handover）

> **STATUS**: 🟢 **ACTIVE（リポジトリ初期化・教材docs分割完了 ✅ ／ 次回セッションは Phase 0 Week 0-1 環境構築・micrograd読解からスタート）**  
> **担当スコープ**: 28週カリキュラム（[`docs/`](file:///Users/sasaki/Desktop/progDoc/LLM-learning/docs) 配下 Phase 0〜4）の履修進捗、実装課題（`mygrad` 等の自作）、検証実験、日々の学習ログ（[`logs/`](file:///Users/sasaki/Desktop/progDoc/LLM-learning/logs)）管理、セッション間引き継ぎ

---

## 0. セッション引き継ぎスナップショット（Session Handover State）
> ※新しい作業セッションを開始する際は、まずこのセクションを読んで再開すること。

- **最終更新**: 2026-11-01
- **現在のフェーズ**: **Phase 0：数学の実践的最小セットの再構築（Week 0-1 着手前）**
- **直近の完了事項**:
  - GitHub パブリックリポジトリ `kenken-201/LLM-learning` の開設と初期プッシュ。
  - ブレスト教材（`tmp.md`）の無損失分割・配置（ルート [`README.md`](file:///Users/sasaki/Desktop/progDoc/LLM-learning/README.md) ＋ [`docs/`](file:///Users/sasaki/Desktop/progDoc/LLM-learning/docs) 配下10ファイル）。
  - 機械学習・Python開発用の標準 [`.gitignore`](file:///Users/sasaki/Desktop/progDoc/LLM-learning/.gitignore) の整備。
  - セッション進行管理用フレームワーク（[`todos/`](file:///Users/sasaki/Desktop/progDoc/LLM-learning/todos), [`logs/`](file:///Users/sasaki/Desktop/progDoc/LLM-learning/logs), [`src/`](file:///Users/sasaki/Desktop/progDoc/LLM-learning/src)）の立ち上げ。
- **現在進行中 / ブロッカー**:
  - **ブロッカーなし**。次回セッションは Python 開発環境の準備（`uv` または `venv`）から即座に着手可能。
- **⚠️ 試行済み・再試行禁止（Dead Ends / Guardrails）**:
  - **写経・コピペ実装の禁止**: Phase 0 の micrograd や Phase 1 の 300行GPT は、コードの丸写しを厳禁とする。テンソル形状と計算グラフの逆伝播を脳内・図解で理解してから自力で書くこと。
  - **autograd の過度な最適化・C++拡張の禁止**: Phase 0 では「正しさ（`torch.autograd.gradcheck` 相当との一致）」だけを追う。高速化は後のフェーズで扱うため深追いしない。
  - **完璧主義・スタックの禁止（15分ルール厳守）**: 15分詰まったら生成AIに質問する。ただし「自分の仮説（形状・想定挙動）」を必ずプロンプトに含めること。
  - **教材正本（`docs/`）の無断要約・改変禁止**: カリキュラム本文の修正・要約は行わず、疑問や学びは `logs/` または [`todos/knowledge.md`](file:///Users/sasaki/Desktop/progDoc/LLM-learning/todos/knowledge.md) に蓄積すること。
- **🎯 次のセッションの最初の1手（Next Immediate Action）**:
  1. **Python 仮想環境の作成とライブラリ導入**:
     ```bash
     python3 -m venv .venv && source .venv/bin/activate
     pip install numpy torch scikit-learn pytest
     ```
  2. [`docs/phase-0-math.md`](file:///Users/sasaki/Desktop/progDoc/LLM-learning/docs/phase-0-math.md) の **Week 0-1: Day 1**（`karpathy/micrograd` の読解：`Value` クラス、計算グラフ構築、`_backward` メカニズム）に着手する。
  3. 初日の学習記録を [`logs/`](file:///Users/sasaki/Desktop/progDoc/LLM-learning/logs) にテンプレートを用いて起票する。

---

## 1. 学習・開発の運用ルール（Roadmap Ground Rules）

1. **1日1コミット・1ログ記録**: コードでもメモでも可。学習ログ（付録C・[`logs/template.md`](file:///Users/sasaki/Desktop/progDoc/LLM-learning/logs/template.md)）に「やったこと・わかったこと・わからなかったこと・AIへの問い」を必ず記録する。
2. **コード → 数式 → 概念の往復**: 抽象的な数式から入らず、「テンソルのshapeはどう変化したか？」「どの次元を潰したか？」から入り、PyTorchの1行として再解釈する。
3. **推論コストを常に意識する**: 単に「学習が回る」だけでなく、スループット（tokens/s）、KV cache、メモリ使用量を常に把握する。
4. **計算資源の予算意識**: クラウドGPUは時間単価で計算し、1実験あたりの上限額を先に設定して暴走を防ぐ（例: 1実験 $5上限）。

---

## 2. フェーズ別進捗チェックリスト（Phase Checklist）

### 🟡 Phase 0：数学の実践的最小セットの再構築（予定期間：3週間）
- [ ] **Week 0-1: 自動微分を自作する（micrograd）**
  - [ ] Day 1: `micrograd` 読解（計算グラフとトポロジカルソート）
  - [ ] Day 2: `Value` クラス自作（スカラー値の四則演算と逆伝播）
  - [ ] Day 3: `tanh`, `exp`, `relu` の追加
  - [ ] Day 4: 2層MLPの自作と勾配降下法
  - [ ] Day 5: 勾配チェック自作（`gradcheck` 相当での数値検証）
  - [ ] 週末: 全体図解と生成AIによるレビュー
  - [ ] **実装課題 0-A**: `mygrad` で `make_moons` 分類（精度85%以上、numpyのみ使用）
- [ ] **Week 0-2: 線形代数・確率をテンソル演算として**
  - [ ] Day 1: ブロードキャストルールの完全言語化
  - [ ] Day 2: `einsum` マスター（matmul, batch matmul, attention）
  - [ ] Day 3: 行列積の逆伝播を手計算・テンソル形状追跡
  - [ ] Day 4: Softmax の数値安定性（Max-trick）と Cross Entropy 勾配導出
  - [ ] Day 5: Multi-head Attention を `einsum` のみで自作
  - [ ] 週末: テンソル形状クイズの実施
  - [ ] **実装課題 0-B**: `einsum` のみで Multi-head Attention 実装＆テスト
- [ ] **Week 0-3: 最適化・数値安定性・初期化**
  - [ ] Day 1: SGD with Momentum 自作
  - [ ] Day 2: AdamW 自作（bias correction, decoupled weight decay）
  - [ ] Day 3: 重み初期化の比較（Xavier, He, std=0.02）
  - [ ] Day 4: Mixed Precision（FP16/BF16）と GradScaler の仕組み理解
  - [ ] Day 5: 学習率スケジュール（Cosine with Warmup）自作
  - [ ] 週末: 2層MLPで最適化手法比較実験
  - [ ] **実装課題 0-C**: AdamW と Cosine Warmup を自作し検証
- [ ] **Phase 0 移行条件の達成判定**:
  - [ ] 1. micrograd 相当を自作し moons 精度85%以上
  - [ ] 2. テンソル演算のブロードキャストを脳内展開可能
  - [ ] 3. Multi-head Attention を einsum で実装可能
  - [ ] 4. AdamW 更新式をコードで書ける
  - [ ] 5. 学習率発散のメカニズムを勾配ノルムで説明できる

### ⚪ Phase 1：Transformerの核心理解（予定期間：5週間）
- [ ] Week 1-1: minGPT 読解
- [ ] Week 1-2: nanoGPT 読解
- [ ] Week 1-3: 自作GPT（写経禁止・300行）
- [ ] Week 1-4: 近代化（RMSNorm, SwiGLU, RoPE）
- [ ] Week 1-5: 推論最適化とベンチマーク（KV cache, tokens/s）

### ⚪ Phase 2：完全オープンOSSでの再現体験（予定期間：6週間）
- [ ] Week 2-1〜2-6: OLMo / SmolLM 読解、データパイプライン自作、100M事前学習、評価レポート

### ⚪ Phase 3：プロダクションレシピの読解（予定期間：5週間）
- [ ] Week 3-1〜3-5: 並列化（DP/TP/PP/ZeRO）、Nemotron, Llama 3, DeepSeek-V3, ポストトレーニング

### ⚪ Phase 4：独自実験とデプロイ（予定期間：8週間＋継続）
- [ ] Week 4-1〜4-8: SFT, LoRA/QLoRA, DPO, GRPO, 量子化, vLLMデプロイ, 公開
