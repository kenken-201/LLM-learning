# Phase 4：独自実験とデプロイ（8週間＋継続）

## 4.1 フェーズの目的と到達基準

**目的**：**自分で問いを立て、実験し、コストを制御して公開する**。ここがプロの入口。前Phaseまでで蒔いた「問いの種」を回収する。

**到達基準**
1. 自作モデル（Phase 2の100M、またはHFの小型モデル）に対して **SFT → 選好学習（DPO or GRPO）** を実施し、**before/after を定量的に比較**できる。
2. **量子化（GPTQ/AWQ/GGUF）** を行い、精度劣化とスループット/メモリのトレードオフ表を作れる。
3. **vLLM** でAPIサーバを立て、**連続バッチング・prefix caching** の効果を測定できる。
4. **コスト試算表**（$ / 1M tokens、$ / 学習1回）を提出できる。
5. 自分の実験を**再現可能な形で公開**（コード＋設定＋ログ＋モデルカード）。
6. 「次の問い」を3つ以上、実験計画付きで書ける。

---

## 4.2 週単位スケジュール

| 週 | テーマ | 時間 | 主教材 |
|---|---|---|---|
| **4-1** | SFT（chat template, masking, packing） | 12h | `huggingface/trl`（`SFTTrainer`）、`axolotl`、Tülu 3のSFT章 |
| **4-2** | LoRA / QLoRA（パラメータ効率的微調整） | 12h | LoRA(2106.09685)、QLoRA(2305.14314)、`unsloth` |
| **4-3** | 選好学習：DPO | 12h | DPO(2305.18290)、`trl` の `DPOTrainer` |
| **4-4** | 選好学習：GRPO / RLVR | 12h | DeepSeekMath §4、`trl` の `GRPOTrainer` |
| **4-5** | 量子化（GPTQ/AWQ/GGUF） | 12h | `AutoGPTQ`, `AutoAWQ`, `llama.cpp` |
| **4-6** | 推論最適化とデプロイ（vLLM） | 12h | `vllm-project/vllm`、PagedAttention(2309.06180) |
| **4-7** | 評価・ベンチマーク・コスト試算 | 12h | `lm-evaluation-harness`、`vllm` の benchmark |
| **4-8** | 公開とレポート、次の問い | 12h | — |

---

## 4.3 日単位タスク例

### Week 4-1：SFT

| Day | タスク |
|---|---|
| **Day1** | 【読む】Tülu 3 (arXiv:2411.15124) §3：SFTデータの構成（persona-driven、汎用/数学/コード/安全のカテゴリ別） |
| **Day2** | 【読む】`trl` の `SFTTrainer`（`trainer/sft_trainer.py`）：`dataset_text_field`, `packing`, `max_seq_length`, `DataCollatorForCompletionOnlyLM`（**prompt部分をlossから除外する仕組み**） |
| **Day3** | 【書く】chat template（`tokenizer.apply_chat_template`）を理解し、**prompt部分に`-100`を立てるcollator**を自作 |
| **Day4** | 【走らせる】小型モデル（例：`HuggingFaceTB/SmolLM2-135M` など）で、自作の小さな指示データ（100〜500件）でSFT。lossと生成品質を記録 |
| **Day5** | 【比較】packing有無、prompt masking有無で、lossと生成品質を比較（**lossの値は比較不能になる点に注意**：maskingするとlossの意味が変わる） |
| **週末** | 【評価】SFT前後で、固定プロンプト20件の応答を並べて目視評価＋簡易ルーブリック |

### Week 4-2：LoRA / QLoRA

| Day | タスク |
|---|---|
| **Day1** | 【読む】LoRA (arXiv:2106.09685) §4（低ランク分解、`B@A` の初期化、`alpha/r` スケーリング） |
| **Day2** | 【読む】QLoRA (arXiv:2305.14314) §3（NF4量子化、double quantization、paged optimizer） |
| **Day3** | 【書く】LoRA層を自作（`nn.Linear` をラップし、`A: (r, in)`, `B: (out, r)`, `scaling = alpha/r`）。**元の重みを凍結** |
| **Day4** | 【走らせる】`peft` でLoRAを適用し、`r=8/16/64`, `target_modules=["q_proj","v_proj"]` vs 全部、で比較 |
| **Day5** | 【走らせる】QLoRA（4bit）で、メモリ削減量と速度・精度を測定 |
| **週末** | 【表】LoRA vs 全パラメータFT vs QLoRA の「メモリ / 時間 / 精度」比較表 |

### Week 4-3：DPO

| Day | タスク |
|---|---|
| **Day1** | 【読む】DPO (arXiv:2305.18290) §4（損失関数の導出）。**参照モデル `π_ref` が必要な理由** |
| **Day2** | 【書く】DPO損失を自作（`logprob` の計算：`log_softmax` → `gather` → `mask` → `sum`） |
| **Day3** | 【書く】選好データ作成：自作SFTモデルで同一プロンプトから2応答を生成し、**ルールベース or 小型reward modelで選別**（人手は最小限） |
| **Day4** | 【走らせる】`trl` の `DPOTrainer` でDPO。`beta=0.1` で開始。**reward margin（chosen - rejected）が増えるか**を監視 |
| **Day5** | 【分析】DPO後のモデルで「応答長が短くなる/長くなる」現象を確認（**長さバイアス**は実務の重要トピック） |
| **週末** | 【比較】SFT only vs SFT+DPO の応答を、ルーブリック（正確性/簡潔性/指示追従）で20件評価 |

### Week 4-4：GRPO / RLVR

| Day | タスク |
|---|---|
| **Day1** | 【復習】DeepSeekMath §4.1のGRPO目的関数を、コードに1対1対応させる |
| **Day2** | 【書く】**検証可能な報酬**を設計：算数の答え合わせ、コードのユニットテスト、JSON形式チェック |
| **Day3** | 【走らせる】`trl` の `GRPOTrainer` で、**1桁/2桁の足し算タスク**をRL。`num_generations=8`, `temperature=1.0` など |
| **Day4** | 【分析】報酬の平均、応答長、正答率の推移。**報酬ハッキング**（例：長く書くだけで報酬が上がる）の兆候を探す |
| **Day5** | 【書く】報酬関数を改善（長さペナルティ、形式ボーナス）し、再実験 |
| **週末** | 【レポート】「RLVRが効く条件」を自分の実験結果から3つ書く |

### Week 4-5：量子化

| Day | タスク |
|---|---|
| **Day1** | 【読む】GPTQ (arXiv:2210.17323) のAbstract + §3（層ごとの量子化）、AWQ (arXiv:2306.00978) §3（活性値の外れ値に着目） |
| **Day2** | 【走らせる】`AutoGPTQ` または `llm-compressor` でGPTQ量子化（4bit）。**キャリブレーションデータ**の選び方を変えて比較 |
| **Day3** | 【走らせる】AWQで量子化。GPTQとの perplexity とスループット比較 |
| **Day4** | 【走らせる】`llama.cpp` でGGUFに変換（`convert_hf_to_gguf.py`）、`Q4_K_M` などで量子化し、CPU推論の速度を測定 |
| **Day5** | 【書く】量子化前後のタスク精度（lm-eval-harnessの数ベンチ）を表に。**「どのタスクが劣化しやすいか」**を考察 |
| **週末** | 【表】bf16 / GPTQ-4bit / AWQ-4bit / GGUF-Q4_K_M の「サイズ / 速度 / 精度」比較 |

### Week 4-6：vLLMデプロイ

| Day | タスク |
|---|---|
| **Day1** | 【読む】PagedAttention (arXiv:2309.06180) §4（KV cacheのページング、断片化削減） |
| **Day2** | 【走らせる】`vllm` でOpenAI互換サーバを起動（`vllm serve <model>`）。`curl` で叩く |
| **Day3** | 【測定】`vllm` の `benchmarks/benchmark_serving.py` で、**request rate / concurrency を変えて**スループットとTTFT/TPOTを測定 |
| **Day4** | 【比較】`transformers` の素朴なgenerateとvLLMで、同一条件下のtokens/sを比較 |
| **Day5** | 【測定】prefix caching（`enable_prefix_caching=True`）の効果を、共通システムプロンプトを持つリクエストで測定 |
| **週末** | 【構成】簡単なFastAPIラッパー＋Docker Composeで「ローカルLLM API」を構築 |

### Week 4-7：評価とコスト

| Day | タスク |
|---|---|
| **Day1** | 【走らせる】`lm-evaluation-harness` で、SFT前/後/DPO後/量子化後を同一条件で評価 |
| **Day2** | 【書く】評価結果を表にまとめ、**統計的不確実性**（サンプル数、信頼区間）に言及 |
| **Day3** | 【書く】**コスト試算表**：①学習コスト（GPU時間 × 単価）②推論コスト（$/1M tokens、モデル別）③量子化による削減率 |
| **Day4** | 【書く】レイテンシ内訳（prefill vs decode、TTFT vs TPOT）を図示 |
| **Day5** | 【レビュー】全実験を「再現手順書」として整理（環境、seed、コマンド、設定ファイル） |
| **週末** | 【公開準備】モデルカード作成（用途、制限、訓練データ、評価結果、倫理的注意） |

### Week 4-8：公開と次の問い

| Day | タスク |
|---|---|
| **Day1** | 【公開】GitHubリポジトリ整備（README、`requirements.txt`、`Makefile`、`configs/`、`scripts/`） |
| **Day2** | 【公開】HF Hubにモデルとデータセットをアップロード（`huggingface_hub`）。モデルカードを充実 |
| **Day3** | 【執筆】技術レポート（ブログ形式でよい）：背景、手法、結果、失敗、コスト、限界 |
| **Day4** | 【執筆】「次の問い」を3つ、それぞれ「仮説・実験設計・必要コスト」付きで書く |
| **Day5** | 【振り返り】Phase 0〜4の学習ログを通読し、**「できるようになったこと」を箇条書き50個**にする |
| **週末** | 【継続計画】次の8週間の実験計画を立てる（＝Phase 4のループを回し続ける） |

---

## 4.4 使用OSS・教材と「読む順番」

| OSS | URL | 読む場所 |
|---|---|---|
| TRL | https://github.com/huggingface/trl | ①`trl/trainer/sft_trainer.py` ②`trl/trainer/dpo_trainer.py`（`dpo_loss`）③`trl/trainer/grpo_trainer.py`（`compute_loss`） |
| PEFT | https://github.com/huggingface/peft | `peft/tuners/lora/layer.py`（`LoraLayer.update_layer`, `forward`） |
| Axolotl | https://github.com/axolotl-ai-cloud/axolotl | `examples/` のYAML、`src/axolotl/core/trainers/` ※要確認 |
| Unsloth | https://github.com/unslothai/unsloth | `unsloth/models/llama.py`（カーネル融合の説明）、READMEのベンチマーク |
| llm-compressor | https://github.com/vllm-project/llm-compressor | `examples/` のGPTQ/AWQレシピ |
| AutoAWQ | https://github.com/casper-hansen/AutoAWQ | `examples/` |
| llama.cpp | https://github.com/ggml-org/llama.cpp | ①`convert_hf_to_gguf.py` ②`examples/server/` ③`docs/` の量子化タイプ一覧 |
| vLLM | https://github.com/vllm-project/vllm | ①`docs/` の "Quickstart" ②`benchmarks/benchmark_serving.py` ③`vllm/core/` のscheduler（読むだけ） |
| lm-evaluation-harness | https://github.com/EleutherAI/lm-evaluation-harness | `README.md`、`lm_eval/models/vllm_causallms.py` |
| 論文 | LoRA 2106.09685 / QLoRA 2305.14314 / DPO 2305.18290 / PagedAttention 2309.06180 / GPTQ 2210.17323 / AWQ 2306.00978 | 各Abstract + 手法Sectionのみ |

---

## 4.5 生成AI活用プロンプト集（Phase 4）

### ✅ 良い vs ❌ 悪い

| ❌ 悪い | ✅ 良い |
|---|---|
| 「DPOのコードを書いて」 | 「私のDPO損失実装（貼付）で、`logprob` を計算するとき `attention_mask` を掛け忘れている可能性がある。`attention_mask` が0の位置のトークンがlossに混入する具体的なケースを構成し、修正版との数値差を示してください」 |
| 「vLLMの使い方」 | 「vLLMのサーバで、同一システムプロンプト（2000トークン）を持つ100リクエストを送るとき、prefix caching有効/無効でTTFTがどう変わるか、PagedAttentionのページ共有の仕組みから説明してください。実際に測るべきメトリクスも列挙して」 |
| 「量子化で精度が落ちる」 | 「GPTQ-4bit化で、私のモデルのmath系ベンチだけ5ポイント落ちた（他は1ポイント以内）。原因仮説を『キャリブレーションデータのドメイン不一致』『外れ値チャネル』『group_size』の3つに絞り、それぞれを切り分ける最小実験を設計して」 |

### コピペ用プロンプト（Phase 4）

```

【P4-1: 損失実装のレビュー（敵対的レビュー）】
以下のDPO損失実装を、あなたは「バグを探すレビュアー」として査読してください。
観点: (1)マスクの扱い (2)log_softmaxの軸 (3)参照モデルのevalモード (4)betaのスケール (5)数値安定性。
各観点で「バグがあるか / ないか / 判断できない」を明示し、判断できない場合は確認コードを書いてください。

```python
（貼付）
```

```

```

【P4-2: 報酬ハッキングの予測】
次のGRPO報酬関数（貼付）を最適化すると、モデルはどんな「ずる」を覚える可能性がありますか？
5つ予測し、それぞれ「検出するための評価指標」と「報酬関数の修正案」を書いてください。

```

```

【P4-3: 推論コストの分解】
私のvLLM測定結果（TTFT=320ms, TPOT=18ms, 出力200トークン, 入力1500トークン, 同時100リクエスト）から、
(1)リクエストあたりのレイテンシ (2)スループット(tokens/s) (3)ボトルネック（prefillかdecodeか）
を計算し、改善の優先順位を3つ提案してください。計算式を明示すること。

```

```

【P4-4: 実験計画のレビュー】
私は「LoRAのrankを上げると指示追従が良くなる」という仮説を検証したい。
以下の実験計画（貼付）の問題点を、交絡因子・サンプルサイズ・評価指標の観点で指摘し、
修正版の計画を提案してください。特に「評価指標が主観的」な点をどう定量化するかを含めてください。

```

```

【P4-5: モデルカードの執筆支援】
以下の情報（学習データ、評価結果、コスト、既知の失敗事例：貼付）から、
HuggingFaceのモデルカードを書いてください。含めるべき項目:
用途 / 制限 / 訓練データ / 評価（ベンチと定性） / 倫理的考慮 / 再現手順。
「過大な主張を避ける」トーンで、不明な項目は "Unknown" と明記してください。

````

---

## 4.6 理解度チェック

**自己クイズ**
1. SFTでprompt部分をlossから外す理由は？ 外さないと何が起きる？
2. LoRAの `alpha/r` スケーリングを変えると何が変わる？
3. QLoRAのNF4はなぜ「情報理論的に最適」と言われるか（正規分布仮定）？
4. DPOの `beta` を大きくすると挙動はどう変わる？
5. GRPOで `num_generations` を増やすと何が良くなり、何が高コストになるか？
6. KV cacheのページサイズを大きくすると、断片化と内部断片化はどう変わるか？
7. 「量子化で精度が落ちるタスク」の典型例は？

**失敗時のチェックリスト**
- SFTでlossが下がるのに生成が崩れる → ①chat templateの不一致 ②`eos`の付与漏れ ③prompt maskingのミス ④学習率過大
- DPOでreward marginが増えない → ①beta ②データの選好がノイジー ③参照モデルが同一 ④lr
- GRPOで報酬が上がらない → ①サンプリング温度 ②group内で報酬が全部同じ（分散ゼロ）③報酬関数のバグ
- vLLMが遅い → ①`gpu_memory_utilization` ②`max_model_len` 過大 ③量子化未使用 ④CPUオフロード
- 量子化で壊れる → ①キャリブレーションデータ ②`group_size` ③外れ値層の除外

---

## 4.7 つまずきポイントと対処

| つまずき | 対処 |
|---|---|
| 実験が発散して収拾がつかない | **実験管理表**（日付/仮説/変更点/結果/次）を必ず作る。1実験1変数 |
| 評価が主観的で進まない | **固定プロンプト20件＋ルーブリック3軸（正確性/簡潔性/形式）** を先に決める。可能ならLLM-as-a-judgeも併用（**判定の揺らぎを測る**） |
| コストが読めない | 実験前に「GPU時間 × 単価」を**必ず見積もってログに書く**。超えたら止める |
| 公開が怖い | モデルカードに**制限と失敗事例を書く**。完璧でなくてよい |
| 「次に何をすればいいかわからない」 | Phase 0〜3の学習ログから**「未解決の疑問」を10個**拾い、実験計画に変換する |

## 4.8 このフェーズで「やらなくてよいこと」

- ❌ フルスクラッチのRLHF（PPO + reward model の完全実装）→ DPO/GRPOで十分
- ❌ Kubernetes/本番運用レベルのインフラ
- ❌ マルチGPUでの学習（単一GPUで回る規模に留める）
- ❌ 全ベンチマークの実施（3〜5個で十分）
- ❌ SOTAの追求（**「自分の問いに答える」ことが目的**）
- ❌ Web UIの作り込み（API + curl で十分）
