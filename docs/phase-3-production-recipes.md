# Phase 3：プロダクションレシピの読解（5週間）

## 3.1 フェーズの目的と到達基準

**目的**：**「個人では回せないが、設計思想は盗める」** 超大規模モデルのレシピを読解し、**自分の小規模実験に移植できる形に翻訳する**。ここでのスキルは「読解 → 抽象化 → 小規模検証」。

**到達基準**
1. Nemotron-4 340B / Nemotron-H、Llama 3、DeepSeek-V3、OLMo 2、K2 Think の技術レポートを読み、**「並列化・データ・安定化・ポストトレーニング・推論コスト」の5軸で比較表**を作れる。
2. 各モデルの「**小規模で再現できる工夫**」を3つ以上抽出し、Phase 2の自作モデルに実装できる。
3. TP/PP/DP/EP の違いを図で説明でき、`Megatron-LM` の設定ファイルでどこに対応するか指せる。
4. reasoning model（long CoT + RLVR）の訓練パイプラインを、**GRPOの目的関数レベル**で説明できる。
5. 「この論文のこの数字は、自分の環境では再現不可能」と**コスト根拠付きで言える**。

**移行条件**：比較表＋「小規模移植計画書」（何を移植し、何を捨てるか、根拠付き）。

---

## 3.2 週単位スケジュール

| 週 | テーマ | 時間 | 主教材 |
|---|---|---|---|
| **3-1** | 大規模並列化とMegatron/NeMo | 12h | `NVIDIA/Megatron-LM`, `NVIDIA/NeMo`, ZeRO(1910.02054) |
| **3-2** | Nemotron系（データ・ハイブリッドMamba） | 12h | Nemotron-4 340B tech report(2406.11704)、Nemotron-H(2504.03624) |
| **3-3** | Llama 3 / DeepSeek-V3（アーキと訓練） | 12h | Llama 3 herd(2407.21783)、DeepSeek-V3(2412.19437) |
| **3-4** | OLMo 2の安定化 + ポストトレーニング概観 | 12h | OLMo 2(2501.00656)、Tülu 3、DPO(2305.18290) |
| **3-5** | Reasoning model（K2 Think / RLVR / GRPO） | 12h | **K2 Think（要確認）**、DeepSeekMath(2402.03300) §GRPO、DeepSeek-R1(2501.12948) |

---

## 3.3 日単位タスク例

### Week 3-1：並列化

| Day | タスク |
|---|---|
| **Day1** | 【読む】ZeRO (arXiv:1910.02054) §3（Stage 1/2/3の違い：optimizer state / gradient / parameter の分割） |
| **Day2** | 【読む】`Megatron-LM` の `README.md` と `megatron/core/parallel_state.py`（`initialize_model_parallel`）※**パスは要確認**（頻繁に変わる）。TP/PP/DPのrank配置を図にする |
| **Day3** | 【読む】`Megatron-LM` の `megatron/core/tensor_parallel/layers.py`（`ColumnParallelLinear`, `RowParallelLinear`）。**「なぜColumn→Rowの順に置くとAll-Reduceが1回で済むか」**を図解 |
| **Day4** | 【読む】NeMo の設定（`NeMo/examples/nlp/language_modeling/conf/` のYAML）。`tensor_model_parallel_size`, `pipeline_model_parallel_size` がどこに書かれるか |
| **Day5** | 【書く】自作モデルで「TPを模擬」する：`ColumnParallelLinear`相当を**シングルGPUで分割実行**し、数値が一致することを確認（実装の理解が目的） |
| **週末** | 【図解】DP/TP/PP/EPの違いを1枚の図に。生成AIに「この図の誤りを指摘して」 |

### Week 3-2：Nemotron

| Day | タスク |
|---|---|
| **Day1** | 【読む】Nemotron-4 340B tech report (arXiv:2406.11704)：**データ（合成データ生成パイプライン）** の章を重点。Web crawl → 合成QA/対話データ生成の発想 |
| **Day2** | 【読む】同レポートの**訓練**（TP=8, PP=12, 学習率、batch size、seq len、FP8?）と**評価**。数字を表に抜き出す |
| **Day3** | 【読む】Nemotron-H (arXiv:2504.03624)：**Mamba-Transformerハイブリッド**。`-H` のアーキ図を読み、Attention層とMamba層の比率、推論速度の主張を確認 |
| **Day4** | 【読む】`NVIDIA/NeMo` の Nemotron 関連レシピ（`NeMo/recipes/` 配下、**要確認**）。設定ファイルから「ハイブリッド構成をどう表現するか」を見る |
| **Day5** | 【書く】**小規模移植**：自作GPTに「Attention層の一部をSSM的（または単純な線形RNN的）に置き換える」実験を行い、lossと速度を比較（**完全なMamba実装は不要**、`mamba-ssm` パッケージがあれば試す） |
| **週末** | 【質問→検証】「Nemotronの合成データ生成を1000分の1の規模でやるなら、どう設計する？」→ 出た案を1つ実装 |

> ⚠️ **要確認**：Nemotron系は世代が多い（Nemotron-4 15B/340B, Nemotron-H, Nemotron Nano, Nemotron 3 など）。**どの世代のレポートかを必ず明記**し、モデルカード（HuggingFace `nvidia/Nemotron-*`）と技術レポートの対応を確認すること。

### Week 3-3：Llama 3 / DeepSeek-V3

| Day | タスク |
|---|---|
| **Day1** | 【読む】Llama 3 herd (arXiv:2407.21783) §3（事前学習：データ、15T tokens、学習率、並列化、**FP8訓練**）と §4（ポストトレーニング：SFT → rejection sampling → DPO） |
| **Day2** | 【読む】同 §3.3（**スケーリング則に基づく学習率の予測**：小モデルで最適lrを探して大モデルに外挿）。**これは個人でも再現できる最重要テクニック** |
| **Day3** | 【読む】DeepSeek-V3 (arXiv:2412.19437) §2（MLA: Multi-head Latent Attention、DeepSeekMoE、MTP: Multi-Token Prediction）、§3（FP8訓練） |
| **Day4** | 【書く】**MLAのKV cache削減効果を計算**：通常MHA/GQA/MLAで、T=8192, L=61のときのKV cacheサイズを比較するスクリプトを書く |
| **Day5** | 【書く】**lr転送実験**：自作モデルで 10M / 30M / 100M の最適lrを探索し、`lr ∝ N^{-a}` のaを推定。100Mの予測値と実測を比較 |
| **週末** | 【比較表】Llama3 vs DeepSeek-V3 vs Nemotron-4 の「データ規模・アーキ・並列化・精度」表 |

### Week 3-4：OLMo 2 + ポストトレーニング

| Day | タスク |
|---|---|
| **Day1** | 【読む】OLMo 2 (arXiv:2501.00656) §2（**QK-norm, z-loss, 学習率の安定化**）、§3（データ：Dolma 2、**データ混合比のablation手法**） |
| **Day2** | 【読む】OLMo 2 の「**訓練ログを全公開**」している点（W&Bリンク）。**実際のloss/grad_normの推移を見る**（これが最大の教材） |
| **Day3** | 【読む】Tülu 3（arXiv:2411.15124）：SFT → DPO → RLVR のパイプライン。「**データの作り方（persona-driven）**」に注目 |
| **Day4** | 【読む】DPO (arXiv:2305.18290) §4（目的関数の導出）。**「報酬モデルなしでRLHF相当」**の意味を数式で |
| **Day5** | 【書く】DPO損失を15行で実装（`logits` から `log_prob` を計算し、`-logsigmoid(β * ((logπ(y_w) - logπ_ref(y_w)) - (logπ(y_l) - logπ_ref(y_l))))`） |
| **週末** | 【計画】「自作モデルにDPOを適用するなら、選好データをどう作るか」を500字で |

### Week 3-5：Reasoning model

| Day | タスク |
|---|---|
| **Day1** | 【確認】**K2 Think の公開情報を自分で確認する**（HF model card、arXiv、公式ブログ）。モデルサイズ、ベースモデル、訓練手法（RLVR/GRPO系）、公開されているコード・データの範囲を**要確認事項として表にまとめる**。⚠️ **本書執筆時点で公開情報が流動的なため、推測で断定しない** |
| **Day2** | 【読む】DeepSeekMath (arXiv:2402.03300) §4（GRPO：group relative policy optimization）。**advantage を group 内で正規化する**アイデアを数式で |
| **Day3** | 【読む】DeepSeek-R1 (arXiv:2501.12948)：RLVR（verifiable rewards：数学の答え合わせ、コードのテスト）によるlong CoT獲得。**「報酬が検証可能」なタスク設計**が本質 |
| **Day4** | 【書く】**GRPOの最小実装**：`prompt → G個サンプリング → reward計算（正解一致で1/0）→ group内で正規化 → policy gradient loss`。**数式1行ずつコードに対応させる** |
| **Day5** | 【走らせる】自作モデル（または小型HFモデル）に、**1桁の足し算タスク**でGRPOを適用し、正答率が上がるか確認 |
| **週末** | 【レポート】「reasoning modelの訓練で、個人が再現できる部分/できない部分」をコスト根拠付きで整理 |

---

## 3.4 使用OSS・論文と「読む順番」

| 対象 | URL | 読む順番・場所 |
|---|---|---|
| Megatron-LM | https://github.com/NVIDIA/Megatron-LM | ①`README.md` ②`megatron/core/parallel_state.py` ③`megatron/core/tensor_parallel/layers.py` ④`megatron/core/pipeline_parallel/` ※**パスは要確認**（v0.10前後で `megatron/` → `megatron/core/` に再編） |
| NeMo | https://github.com/NVIDIA/NeMo | `examples/` 配下の言語モデルレシピ、`nemo/collections/llm/` ※要確認 |
| Nemotron-4 340B | https://arxiv.org/abs/2406.11704 | §2（データ）, §3（訓練）, §4（評価） |
| Nemotron-H | https://arxiv.org/abs/2504.03624 | §2（アーキ）, §3（訓練） |
| Llama 3 | https://arxiv.org/abs/2407.21783 | §3（事前学習）, §3.3（スケーリング則）, §4（ポストトレーニング） |
| DeepSeek-V3 | https://arxiv.org/abs/2412.19437 | §2（MLA, MoE, MTP）, §3（FP8） |
| DeepSeekMath (GRPO) | https://arxiv.org/abs/2402.03300 | §4.1（GRPO目的関数） |
| DeepSeek-R1 | https://arxiv.org/abs/2501.12948 | §2（RL）, §3（失敗と教訓） |
| OLMo 2 | https://arxiv.org/abs/2501.00656 | §2（安定化）, §3（データ混合） |
| Tülu 3 | https://arxiv.org/abs/2411.15124 | §3（SFT）, §4（DPO）, §5（RLVR） |
| DPO | https://arxiv.org/abs/2305.18290 | §4（目的関数） |
| ZeRO | https://arxiv.org/abs/1910.02054 | §3, §5-7 |
| **K2 Think** | **要確認**：HuggingFace `MBZUAI/K2-Think*`、MBZUAI公式ブログ、arXiv | モデルカードで「ベースモデル・訓練手法・公開範囲」を確認してから読む。**公開コードがあるかを最初に確認** |

---

## 3.5 生成AI活用プロンプト集（Phase 3）

### ✅ 良い vs ❌ 悪い

| ❌ 悪い | ✅ 良い |
|---|---|
| 「Nemotronの論文を要約して」 | 「Nemotron-4 340Bのtech report §3の訓練設定から、(1)global batch size (2)学習率 (3)並列構成 (4)seq len を数値で抜き出し、『Llama 3の対応する値との比率』を表にしてください。数値が本文にない場合は"記載なし"と明記し、推測しないでください」 |
| 「GRPOを教えて」 | 「DeepSeekMathのGRPO目的関数（数式貼付）を、PyTorchの擬似コードに1対1対応させてください。各行の右に数式のどの項かをコメントで書いてください。特に『group内正規化』がどの行か明示して」 |
| 「Megatronが難しすぎる」 | 「私の目的は『TP=2でColumnParallelLinearとRowParallelLinearをどう配置するとAll-Reduceが1回で済むか』を理解すること。`parallel_state.py`と`layers.py`の該当行だけを抜き出し、通信が発生する箇所に★を付けて、forwardの流れを図解してください」 |

### コピペ用プロンプト（Phase 3）

```

【P3-1: 数値の抽出と検算（ハルシネーション防止）】
以下の論文本文（貼付）から数値のみを抽出して表にしてください。
ルール: (1)本文に明記された数値のみ (2)単位を必ず付記 (3)「約」「最大」などの修飾語も保持
(4)記載がない項目は「記載なし」と書く（推測禁止）。
抽出項目: パラメータ数 / 学習トークン数 / バッチサイズ / 学習率 / 系列長 / GPU数 / 訓練日数。

```

```

【P3-2: 小規模移植の設計】
Nemotron-4の「合成データ生成パイプライン」（論文§2の記述: 貼付）を、
個人の予算（$50、1週間）で1000分の1スケールで再現する計画を立ててください。
(a)教師モデルの選定 (b)生成するタスク種別 (c)フィルタリング (d)検証方法 を含め、
各ステップのコスト見積もりと「省略可能な部分」を明示してください。

```

```

【P3-3: コスト妥当性の判定（プロの必須スキル）】
次の主張の妥当性を、計算根拠付きで評価してください:
「Llama 3の15Tトークン事前学習を、A100 40GBを8枚スポット（$1.2/h/枚）で再現する」
必要なGPU時間、総コスト、実現可能性（yes/no）と、noの場合の代替案を提示してください。
前提（FLOPS利用率など）は明示し、不確実な部分はレンジで示してください。

```

```

【P3-4: アーキテクチャの目的逆算】
DeepSeek-V3のMLAについて、(1)解決したい問題 (2)MHA/GQAとの違い (3)推論時のKV cache削減量
を、T=8192, L=61, d=7168, n_head=128 の具体値で計算してください。
さらに「なぜGQAでは不十分でMLAが必要だったか」を、精度とメモリのトレードオフで説明してください。

```

```

【P3-5: 論文の"読まなくていい部分"の特定】
私は「reasoningモデルのRL訓練で、報酬設計とデータ選別の勘所」を知りたい。
DeepSeek-R1の論文（貼付）のうち、読むべきSectionと飛ばすべきSectionを理由付きで分類し、
読むべきSectionについて「理解すべき問い」を3つずつ立ててください。

```

---

## 3.6 理解度チェック

**説明できるべき概念**
1. DP/TP/PP/EP の違いと、それぞれが何を分割するか
2. ZeRO Stage 1/2/3 の違い
3. MLAが「GQAよりKV cacheを減らせる」理由
4. FP8訓練で必要になる工夫（スケーリング、master weights）
5. GRPOのadvantage計算がPPOとどう違うか
6. 「検証可能な報酬」が使えるタスクと使えないタスク
7. 小モデルで最適lrを探し大モデルに外挿する手法の前提条件

**失敗時のチェックリスト**
- 論文の数字が自分の環境で再現できない → **それが正常**。レポートには「再現不能である理由の定量化」を書く
- 用語が多すぎて追えない → **5軸表**（並列/データ/安定化/ポスト/推論）に常に書き戻す
- 数式が読めない → 目的関数だけに絞り、`for`文に翻訳する

---

## 3.7 つまずきポイントと対処

| つまずき | 対処 |
|---|---|
| 論文が多すぎて挫折 | **1週1テーマ、1テーマ1論文**。比較表に書けない情報は捨てる |
| 「自分には関係ない世界」と感じる | **必ず「小規模移植」を1つ実施**（例：lr転送、GRPOの足し算、合成データ1000件） |
| Megatronのコードが難解 | 目的を「通信箇所の特定」に限定。全部読まない |
| 数値のハルシネーション | プロンプトP3-1で「記載なし」を強制。**論文PDFを必ず貼る** |
| K2 Think等の情報が錯綜 | **一次情報（model card / arXiv / 公式ブログ）のみ信頼**。ブログ記事の数字は使わない |

## 3.8 このフェーズで「やらなくてよいこと」

- ❌ 340B級モデルの訓練実行（**読む対象**）
- ❌ Megatron-LMの全コード読解（通信箇所だけでよい）
- ❌ MoE の完全実装（概念と損失の理解で十分）
- ❌ FP8訓練の実装（読むだけ。Phase 4でbf16を使う）
- ❌ 全論文の精読（**該当Sectionのみ**）
- ❌ マルチノード環境の構築
