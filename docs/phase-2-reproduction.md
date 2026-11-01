# Phase 2：完全オープンOSSでの再現体験（6週間）

## 2.1 フェーズの目的と到達基準

**目的**：「自分で書いた小さなGPT」から「**プロの訓練パイプライン**」への橋渡し。OLMo（完全オープン：データ・コード・ログ・中間チェックポイントまで公開）とSmolLM（小型モデルの実践的レシピ）を読み、**自分のデータで100M級モデルを事前学習する**。

**到達基準**
1. OLMoの `olmo/train.py` の訓練ループを読み、nanoGPTの `train.py` との**差分を10項目以上**列挙できる。
2. `MemMapDataset` ベースのデータパイプライン（tokenize → packing → memmap → 学習時slice）を**自作できる**。
3. FineWeb-Edu（またはDCLM）のサブセットを使い、**100Mパラメータ級モデルを8〜12 GPU時間以内**で訓練し、val lossと下流評価（lm-eval-harness の tiny bench）を出す。
4. 学習ログ（loss, lr, grad_norm, throughput）から**「データ品質 vs モデル容量 vs 学習不足」を切り分ける**ことができる。
5. トークナイザ（BPE）の学習と語彙数変更がlossに与える影響を説明できる。

**移行条件**：3の成果物（訓練スクリプト＋ログ＋評価レポート）をGitHubに公開。

---

## 2.2 週単位スケジュール

| 週 | テーマ | 時間 | 主教材 |
|---|---|---|---|
| **2-1** | OLMo：モデルと設定の読解 | 12h | `allenai/OLMo`（`olmo/model.py`, `olmo/config.py`） |
| **2-2** | OLMo：訓練ループとデータパイプライン | 12h | `olmo/train.py`, `olmo/data/`, `scripts/`、Dolma |
| **2-3** | SmolLM：小型モデルの実践レシピ | 12h | `huggingface/smollm`、SmolLM2論文(2502.02737) |
| **2-4** | データパイプライン自作＋トークナイザ | 12h | FineWeb-Edu / Cosmopedia、`tokenizers` |
| **2-5** | 100Mモデル事前学習（実行週） | 12h+GPU | 自作パイプライン |
| **2-6** | 評価・分析・レポート | 12h | `EleutherAI/lm-evaluation-harness`、W&B |

---

## 2.3 日単位タスク例

### Week 2-1：OLMoモデル読解

| Day | タスク |
|---|---|
| **Day1** | 【読む】`allenai/OLMo` の `README.md` とリポジトリ構成（`olmo/`, `scripts/`, `configs/`）。**OLMo 2以降は `allenai/OLMo-core` に移行している可能性があるため要確認**（README記載の推奨リポジトリを確認） |
| **Day2** | 【読む】`olmo/config.py`：`ModelConfig`（`d_model`, `n_heads`, `n_layers`, `mlp_hidden_size`, `rope_theta`, `attention_layer_norm`, `include_bias` など）。nanoGPTのconfigとの差分を表に |
| **Day3** | 【読む】`olmo/model.py`：`OLMoBlock`（QK-normの有無、`attention_layer_norm`）、`OLMo.__init__`、`forward`（`attention_bias`, `alibi` 等の分岐）。**`OLMoBlock` と nanoGPT `Block` の差分を10行でメモ** |
| **Day4** | 【読む】`olmo/model.py` の `BufferCache`（KV cacheのリングバッファ実装）。自作KV cacheとの差分を理解 |
| **Day5** | 【書く】自作GPTに**QK-norm**と**z-loss**を実装（OLMo 2の安定化策）。小規模でloss曲線が安定するか比較 |
| **週末** | 【質問→メモ】「OLMoがnanoGPTに比べて"訓練が安定する"ために入れている工夫を、コード根拠付きで列挙して」（AIに聞いた後、**必ずコードで裏取り**） |

### Week 2-2：OLMo訓練ループとデータ

| Day | タスク |
|---|---|
| **Day1** | 【読む】`olmo/train.py`：`Trainer` クラス、`train_step`、`get_labels`（`-100` マスク）、`cross_entropy_loss`（z-lossオプション）、grad accumulation |
| **Day2** | 【読む】`olmo/train.py`：checkpoint保存、`save_interval`、`save_interval_unsharded`、再開ロジック。**「壊れても再開できる」設計**を学ぶ |
| **Day3** | 【読む】`olmo/data/__init__.py`：`MemMapDataset`（`__getitem__` が `np.memmap` からsliceする仕組み）、`pack_dataset`（複数文書を連結して `block_size` に詰める）、`TokenizedDataset` |
| **Day4** | 【読む】`scripts/`（`train.py` の起動引数）と `configs/`（例：`OLMo-1B`設定）。**どの引数が実務で最重要か**（lr, warmup, global batch size, seq len, z-loss係数）を抜き出す |
| **Day5** | 【読む】Dolma（https://github.com/allenai/dolma）：データの構成（CommonCrawl, C4, コード, 数学, 論文…）とフィルタリング方針。**「何を入れないか」の設計思想**に注目 |
| **週末** | 【書く】OLMoのtrain loopを参考に、自作train.pyに「resume機能」と「grad accumulation」を追加 |

> ⚠️ **要確認**：OLMoは2024→2025でリポジトリ構成が大きく変化（`OLMo` → `OLMo-core`）。ファイルパスは**手元で `git clone` して `find . -name "*.py" | head -50` で確認**してから読むこと。本書のパスはOLMo 1系を基準にしている。

### Week 2-3：SmolLM

| Day | タスク |
|---|---|
| **Day1** | 【読む】`huggingface/smollm` の `README.md`、ディレクトリ構成（`smollm/`, `smollm2/`, `smollm3/`, `textbooks/` など）。**どのモデルがどの実装か要確認** |
| **Day2** | 【読む】SmolLM2論文(arXiv:2502.02737)：データ（FineWeb-Edu, DCLM, Cosmopedia-v2, SmolTalk）、4段階訓練（pretrain → midtrain → SFT → DPO）、**「小型モデルはデータ品質がすべて」**という主張 |
| **Day3** | 【読む】`smollm/` の訓練スクリプト（nanotronベース）。config YAML（`135M`, `360M`, `1.7B`）を見て、**パラメータ数とハイパラの対応**を表にする |
| **Day4** | 【読む】SmolTalk（SFTデータ）の構成。**「事前学習データとSFTデータの役割の違い」**を整理 |
| **Day5** | 【書く】SmolLMのconfigを自作GPTに移植し、135M相当のモデルを定義（`d_model=576, n_layer=30, n_head=9` 等は**要確認**、configを直接読む） |
| **週末** | 【比較表】OLMo vs SmolLM vs nanoGPT の「データ・アーキ・訓練レシピ」3列比較表を作る |

### Week 2-4：データパイプライン自作

| Day | タスク |
|---|---|
| **Day1** | 【走らせる】FineWeb-Edu（https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu）の `sample-10BT` から**1〜2GBだけ**ダウンロード（`datasets` の `streaming=True` で必要分だけ） |
| **Day2** | 【書く】テキスト正規化（HTML除去は既に済み、`unicodedata.normalize`、空白処理）と**重複除去（MinHashは過剰。まずは完全一致ハッシュ）** |
| **Day3** | 【書く】BPEトークナイザを `tokenizers` で学習（語彙 32k or 50k）。**自作GPT用に `vocab_size` を合わせる** |
| **Day4** | 【書く】tokenize → `np.memmap` で `train.bin` / `val.bin` に書き出し（`dtype=np.uint16`）。`pack_dataset`（EOS挟み込み、`block_size` で切る） |
| **Day5** | 【書く】`MemMapDataset` 相当を自作し、`DataLoader` で読み、**スループット（tokens/s）を測定**。ボトルネックがI/OかGPUかを切り分け |
| **週末** | 【検証】データのサニティチェック：先頭100サンプルをデコードして目視、token分布のヒストグラム、`block_size` に対する文書境界の割合 |

### Week 2-5：100Mモデル事前学習（実行週）

| Day | タスク |
|---|---|
| **Day1** | 【準備】学習率スケジュール決定（warmup 1-2%、cosine）、global batch size（例：0.5M tokens/step）、grad accumulation。**Chinchilla則（arXiv:2203.15556）で「モデルサイズ vs トークン数」の目安を計算**（100Mなら約2Bトークンが目安。今回は**計算予算優先で0.5〜1Bトークンに縮小**し、その旨をレポートに明記） |
| **Day2** | 【走らせる】短時間（30分）の試運転：lossが下がるか、throughput、メモリ、nanチェック。**この時点で`torch.compile`とAMPを有効化** |
| **Day3** | 【走らせる】本訓練開始（spot A100 40GB × 8〜12h想定）。W&Bで監視。**チェックポイントを30分ごとに保存**（spot切られ対策） |
| **Day4** | 【監視】loss曲線、grad_norm、lr。異常時の判断基準（下記チェックリスト） |
| **Day5** | 【完了】訓練終了、最終ckptとログを保存。`sample.py` 相当で生成品質を目視確認 |
| **週末** | 【分析】loss曲線を「warmup期 / 線形減少期 / 飽和期」に分け、それぞれの解釈をレポートに |

### Week 2-6：評価とレポート

| Day | タスク |
|---|---|
| **Day1** | 【読む】`EleutherAI/lm-evaluation-harness` のREADME、`lm_eval --model hf --model_args pretrained=...` の使い方 |
| **Day2** | 【走らせる】自作モデルをHF形式に変換（`transformers` の `GPT2LMHeadModel` 相当、またはカスタム `PreTrainedModel`）し、`lambada_openai`, `hellaswag`, `arc_easy` を評価（**100Mではほぼチャンスレベル、それでよい**） |
| **Day3** | 【書く】perplexity を val set で計算。**ドメイン別（コード/数学/自然文）に分けた perplexity** を出す |
| **Day4** | 【書く】ablation：データを50%に減らした場合、モデルを半分にした場合のloss差を**予測→検証** |
| **Day5** | 【書く】レポート執筆（構成：目的/手法/計算予算/結果/失敗と学び/次の問い） |
| **週末** | 【公開】GitHubに公開。READMEに再現手順とコストを明記 |

---

## 2.4 使用OSS・教材と「読む順番」

| OSS | URL | 読む順番 |
|---|---|---|
| OLMo | https://github.com/allenai/OLMo | ①`README.md` ②`olmo/config.py` の `ModelConfig` ③`olmo/model.py` の `OLMoBlock`→`OLMo` ④`olmo/train.py` の `train_step`→`Trainer` ⑤`olmo/data/` ⑥`scripts/train.py` |
| OLMo-core | https://github.com/allenai/OLMo-core | **要確認**：OLMo 2系の実装。`olmo_core/nn/transformer/` 配下（パスは要確認） |
| Dolma | https://github.com/allenai/dolma | `README.md` のデータ構成、`docs/` のフィルタリング方針 |
| SmolLM | https://github.com/huggingface/smollm | ①`README.md` ②`smollm/`（または `smollm2/`）の `README` と config YAML ③訓練スクリプト |
| SmolLM2論文 | https://arxiv.org/abs/2502.02737 | §2（データ）, §3（訓練レシピ）, §4（結果） |
| nanotron | https://github.com/huggingface/nanotron | **読むだけ**：`src/nanotron/` の model/pipeline 部分（Phase 3の予習） |
| FineWeb-Edu | https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu | dataset card（フィルタリング基準、教育的スコア） |
| Cosmopedia | https://huggingface.co/datasets/HuggingFaceTB/cosmopedia | dataset card（合成データの作り方） |
| lm-evaluation-harness | https://github.com/EleutherAI/lm-evaluation-harness | `README.md` の "Basic Usage"、`lm_eval/models/huggingface.py` |
| Chinchilla | https://arxiv.org/abs/2203.15556 | §3（スケーリング則）、Figure 1-2 |
| OLMo 2論文 | https://arxiv.org/abs/2501.00656 | §2（アーキ）, §3（訓練の安定化：QK-norm, z-loss） |

---

## 2.5 生成AI活用プロンプト集（Phase 2）

### ✅ 良い vs ❌ 悪い

| ❌ 悪い | ✅ 良い |
|---|---|
| 「OLMoのコードを説明して」 | 「`olmo/data/__init__.py` の `MemMapDataset.__getitem__` が、`np.memmap` からどうやって `block_size` 分を切り出すか、`start`/`end` のインデックス計算を1行ずつ追って。境界（データ末尾）の扱いも説明して」 |
| 「データパイプラインの作り方」 | 「私のtokenizeスクリプト（貼付）は、10GBのテキストで実行するとメモリを食い尽くす。`np.memmap` でストリーミング書き込みする形に書き換えて。バッファサイズとチャンク境界の扱いも明示して」 |
| 「学習がうまくいきません」 | 「100Mモデル、0.8Bトークン、lr 6e-4 cosine、warmup 2000step で、lossが3.2から2.9で停滞。同時刻のgrad_normは0.8前後で安定。データはFineWeb-Edu。この状況で疑うべき順に5つ挙げ、それぞれ"切り分けるための最小実験"を設計して」 |

### コピペ用プロンプト（Phase 2）

```

【P2-1: 差分抽出（リファレンス実装との比較）】
以下2つの訓練スクリプト（A: nanoGPT train.py, B: OLMo train.py の該当部分）を比較し、
差分を「(1)データ (2)最適化 (3)安定化 (4)ロギング (5)再開性」のカテゴリに分類して表にしてください。
各行に「なぜその差分が必要か（大規模化・長時間化・多ノード化のどれに対応するか）」を書いてください。
A: （貼付）
B: （貼付）

```

```

【P2-2: データ品質の診断】
私が訓練した100Mモデルの、ドメイン別val perplexityが以下です:

- 自然文: 18.2 / コード: 6.1 / 数学: 9.8 / 日本語: 42.0
  データ構成はFineWeb-Edu 90% + コード10%（日本語ほぼ0）。
  この結果から「データ構成に起因する」と「モデル容量に起因する」を切り分けるには、
  どんな追加実験をどの順でやるべきか、コスト見積もり付きで提案してください。

```

```

【P2-3: スケーリング則の検算】
Chinchillaの計算最適化（D = 20N）に従うと、N=100Mのとき最適トークン数は？
また、私の予算（GPU 10時間）でD=1Bに達するにはthroughputが何tokens/s必要か、
式と数値で示してください。そのthroughputはA100 40GB 1枚で現実的かも評価してください。

```

```

【P2-4: 失敗ログの解釈】
訓練ログの抜粋（loss, lr, grad_norm の時系列、貼付）を見て、
(a) 学習率が高すぎる (b) データに破損がある (c) モデルが小さい、のどれが最も疑わしいか、
根拠をログの具体的な数値で示してください。断定できない場合は「追加で必要なログ」を挙げてください。

```

```

【P2-5: トークナイザの影響評価】
語彙サイズ32kと50kのBPEで、同じコーパスの「1トークンあたりの平均文字数」と
「系列長の分布」がどう変わるか、計算方法（コード）と、
それが学習コスト（tokens/s × 必要トークン数）に与える影響を説明してください。

```

---

## 2.6 理解度チェック

**自己クイズ**
1. `pack_dataset` はなぜ必要？ パディングだけではダメな理由は？（計算の無駄＋attention maskの複雑化）
2. `-100` ラベルは何を意味する？ 損失計算のどこで無視される？
3. `np.memmap` を使う理由を3つ（メモリ、起動時間、ランダムアクセス）
4. `uint16` で書く制約は？（語彙数 ≤ 65535）
5. QK-norm と z-loss はそれぞれ何を抑えるか？（logitの爆発／logsumexpのドリフト）
6. 100Mモデルで `hellaswag` がチャンスレベルなのはなぜ正常か？

**失敗時のチェックリスト**
- lossが下がらない → ①ラベルのshiftミス（`x[:, :-1]` vs `y[:, 1:]`）②lr ③データが全部同じ ④`-100`だらけ
- lossが急上昇して戻らない → ①lr ②データの異常（特定シャードの破損）③チェックポイント再開時のoptimizer state
- throughputが出ない → ①DataLoaderがボトルネック（`num_workers`, `pin_memory`）②`memmap`のランダムアクセス ③AMP未使用 ④`torch.compile`未使用
- 評価がおかしい → ①chat templateの混入 ②EOSの扱い ③perplexityの計算（平均の取り方：token平均 vs batch平均）

---

## 2.7 つまずきポイントと対処

| つまずき | 対処 |
|---|---|
| OLMoのコードが巨大で圧倒される | **「読むファイルを3つに絞る」**（`config.py`, `model.py` の `OLMoBlock`, `train.py` の `train_step`）。それ以外は無視 |
| データ準備で1週間溶ける | **上限を決める**：「データは2GB、前処理は1日」。品質より「回して学ぶ」を優先 |
| クラウドGPUのコストが怖い | ①spot instance ②チェックポイント30分 ③**最初に30分だけ回して見積もる** ④`nvidia-smi`で利用率監視 |
| 自作モデルがHF形式に変換できない | `state_dict` のキー名対応表を作る。`transformers` の `GPT2LMHeadModel` のキー名を `print(model.state_dict().keys())` で確認 |
| ablationの意味がわからない | **「1回に1変数」**。それ以外は禁止。結果は表に |

## 2.8 このフェーズで「やらなくてよいこと」

- ❌ 10Bトークン級の本格事前学習（予算外。**読む対象**）
- ❌ 分散学習の実装（Phase 3）
- ❌ データの高度な重複除去（MinHash-LSH、SemDedup）
- ❌ 合成データ生成パイプラインの自作（Cosmopediaを読むだけ）
- ❌ SFT/RLHF（Phase 4）
- ❌ モデルのSOTA追求（**再現と分析が目的**）
