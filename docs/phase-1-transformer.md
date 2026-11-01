# Phase 1：Transformerの核心理解（minGPT → nanoGPT）（5週間）

## 1.1 フェーズの目的と到達基準

**目的**：Transformerを「知っている」から「**白紙から書けて、壊れた箇所を形状と数値から特定できる**」に変える。ここでのゴールは**推論コストの理解**を含む。

**到達基準**
1. `nanoGPT/model.py` を見ずに、GPT（Embedding → N×Block(CausalSelfAttention + MLP) → LayerNorm → lm_head）を**300行以内で書ける**。
2. `shakespeare_char` で **val loss 1.5前後**まで到達し、loss曲線から「過学習/学習不足/lr不正」を判別できる。
3. **KV cacheを自作**し、生成トークン数に対するレイテンシの伸びを、cacheなしと比較して**グラフで示せる**。
4. Attentionの計算量 `O(T²d)` とKV cacheのメモリ量 `2 × L × H_kv × T × d_head × bytes` を**暗算できる**。
5. RoPE / RMSNorm / SwiGLU / GQA が「何を解決する部品か」をそれぞれ1分で説明できる。

**移行条件**：上記1・3が達成でき、かつ自作GPTのval lossがnanoGPTと同条件で**±0.05以内**に収まること。

---

## 1.2 週単位スケジュール

| 週 | テーマ | 時間 | 主教材 |
|---|---|---|---|
| **1-1** | minGPT読解：Attentionの最小実装 | 12h | `karpathy/minGPT`、`projects/adder` |
| **1-2** | nanoGPT読解：訓練ループとデータ | 12h | `karpathy/nanoGPT`（`train.py`, `config/`, `data/shakespeare_char/prepare.py`） |
| **1-3** | 自作GPTゼロから（写経禁止） | 12h | 自作 + `nanoGPT/model.py` を答え合わせに使う |
| **1-4** | 近代化：RoPE/RMSNorm/SwiGLU/GQA | 12h | Llama論文(2302.13971) §2、RoPE(2104.09864)、GQA(2305.13245)、`KellerJordan/modded-nanogpt` |
| **1-5** | 推論：KV cache・サンプリング・ベンチ | 12h | `nanoGPT/sample.py`, `bench.py`、`meta-pytorch/gpt-fast` |

---

## 1.3 日単位タスク例

### Week 1-1：minGPT読解

| Day | タスク |
|---|---|
| **Day1** | 【読む】`minGPT/mingpt/model.py`：`NewGELU` → `CausalSelfAttention.__init__`（`c_attn` が `3*n_embd` を一括で出す理由）→ `forward`（`view/transpose` で `(B,T,C)→(B,nh,T,hs)`）。**shapeを紙に書きながら** |
| **Day2** | 【読む】`CausalSelfAttention` の `bias` マスク（`torch.tril`）と `masked_fill`。`F.scaled_dot_product_attention` を使わない版で理解。【質問】「`att = att.masked_fill(bias[:,:,:T,:T]==0, float('-inf'))` の `:T,:T` が必要な理由」 |
| **Day3** | 【読む】`Block`（pre-LN か post-LN か）、`GPT.forward`（`tok_emb + pos_emb` → blocks → `ln_f` → `lm_head`）。【走らせる】`projects/adder/adder.py` をそのまま実行 |
| **Day4** | 【書く】`adder.py` のデータ生成部を読み、**桁数を3→5に増やして**学習を回す。lossの伸びを記録 |
| **Day5** | 【書く】`CausalSelfAttention` を `einsum` で書き直し、数値一致を確認。速度も比較（`torch.cuda.synchronize()` + `time.perf_counter`） |
| **週末** | 【説明会】「Attentionを数式なしで、shapeだけで説明する」をAIに実施→添削 |

### Week 1-2：nanoGPT読解

| Day | タスク |
|---|---|
| **Day1** | 【読む】`nanoGPT/data/shakespeare_char/prepare.py`：char→id、train/val分割、`train.bin/val.bin` への `np.memmap` 書き出し。【質問】「なぜ`.bin`に`uint16`で書くのか、語彙数との関係で説明して」 |
| **Day2** | 【読む】`train.py`：`get_batch`（`ix` ランダムサンプリング、`torch.from_numpy`、`.pin_memory()`、`non_blocking=True`）、`estimate_loss`（`model.eval()` + `torch.no_grad()`） |
| **Day3** | 【読む】`train.py`：`get_lr`（warmup+cosine）、`configure_optimizers`、`grad_clip`、AMP（`torch.amp.autocast`/`GradScaler`）、`gradient_accumulation` |
| **Day4** | 【走らせる】`config/train_shakespeare_char.py` で訓練。**W&B（or TensorBoard）にloss/lr/grad_normを記録**。【書く】`eval_interval` を短くして過学習の兆候を見る |
| **Day5** | 【書く】`config` の `n_layer/n_head/n_embd/block_size/dropout/lr` を**1つずつ**変えて、val lossへの影響を表にする（＝初めての ablation） |
| **週末** | 【読む】`model.py` の `LayerNorm`, `CausalSelfAttention`, `MLP`, `Block`, `GPT` を**Day1-5の知識で再読**。`F.scaled_dot_product_attention` の引数（`is_causal`, `dropout_p`）を確認 |

### Week 1-3：自作GPT（写経禁止）

| Day | タスク |
|---|---|
| **Day1** | 【書く】`mygpt/model.py`：`config` dataclass、`LayerNorm`、`CausalSelfAttention`（SDPA不使用版でまず書く） |
| **Day2** | 【書く】`MLP`、`Block`、`GPT.__init__`（`wte`, `wpe`, blocks, `ln_f`, `lm_head`、weight tying） |
| **Day3** | 【書く】`GPT.forward(idx, targets=None)`：loss計算、`generate`（cacheなし）。【検算】パラメータ数をnanoGPTと一致させる（`GPT-2 small` で124M） |
| **Day4** | 【書く】`train.py` 自作：`get_batch`, `estimate_loss`, `get_lr`, `AdamW`, AMP, grad clip。`shakespeare_char` で訓練開始 |
| **Day5** | 【走らせる】自作 vs nanoGPT の loss曲線比較。【質問】「自作の方がlossが高い。原因候補を、初期化・lr・weight decay・データローダの順に、確認方法付きで列挙して」 |
| **週末** | 【書く】差分修正 → val loss 1.5台到達。**ログに「何が原因だったか」を必ず書く** |

### Week 1-4：近代化（モダン部品）

| Day | タスク |
|---|---|
| **Day1** | 【読む】Llama 1 (arXiv:2302.13971) §2.1-2.2（RMSNorm, SwiGLU, RoPE）。【書く】`RMSNorm` を10行で実装し `LayerNorm` と数値比較 |
| **Day2** | 【読む】RoPE (arXiv:2104.09864) §3.2。【書く】`precompute_freqs_cis` + `apply_rotary_emb` を実装。【質問】「θ_i = 10000^(-2i/d) のとき、d=64でi=0とi=31の波長を具体的な数値で示して」 |
| **Day3** | 【書く】`SwiGLU`（`gate_proj`, `up_proj`, `down_proj`, `F.silu(gate)*up`）。パラメータ数をMLPと揃えるには `hidden = 8/3*d` の調整が必要な理由を確認 |
| **Day4** | 【読む】GQA (arXiv:2305.13245) §2。【書く】`n_kv_head` を分離し、`repeat_interleave` で展開する実装。KV cacheサイズを計算 |
| **Day5** | 【読む】`KellerJordan/modded-nanogpt` の `train_gpt.py`（**Muon optimizer、`resid_lambdas`、`x0` 加算、logit softcap** など）。【質問】「modded-nanogptの高速化テクニックを、"数値安定性/最適化/アーキ/データ"の4分類で整理して」 |
| **週末** | 【書く】自作GPTにRoPE+RMSNorm+SwiGLU+GQAを全部入れ、パラメータ数を揃えた上で**val lossと学習速度**をベースラインと比較 |

> ⚠️ **modded-nanogpt の各テクニックは「要確認」**：リポジトリは活発に更新される（レコード更新のたびにコードが変わる）。読む際は **commit hash を固定**（`git checkout <hash>`）し、READMEの "Record" 履歴と対応させること。

### Week 1-5：推論最適化とベンチマーク

| Day | タスク |
|---|---|
| **Day1** | 【読む】`nanoGPT/sample.py`：`temperature`, `top_k`, `idx_cond`（`block_size` で切る）。【書く】top-p（nucleus）を追加 |
| **Day2** | 【書く】**KV cache実装**：`CausalSelfAttention` に `past_kv` を渡し、`k = cat([past_k, k_new])`。【shape表】cacheありの `k`: `(B, nh, T_past+T_new, hs)` |
| **Day3** | 【走らせる】**ベンチマーク**：cacheなし/ありで、生成トークン数 1,10,50,100,200 のレイテンシを計測し、グラフ化（cacheなしはO(T³)的に伸びる） |
| **Day4** | 【読む】`gpt-fast`（https://github.com/meta-pytorch/gpt-fast）の `generate.py`：`torch.compile`、int8/int4量子化、speculative decoding（`gpt-fast`の `speculate`）。【質問】「speculative decodingが速度向上する条件を、acceptance rate と draft model コストの式で示して」 |
| **Day5** | 【書く】自作モデルに `torch.compile` を適用し、`bench.py` 相当のスループット（tokens/s）を測定。**メモリ使用量（`torch.cuda.max_memory_allocated`）も記録** |
| **週末** | 【成果物】Phase 1レポート：自作GPTのコード＋loss曲線＋スループット表＋KV cacheのメモリ計算式 |

---

## 1.4 使用OSS・教材と「読む順番」

### minGPT（https://github.com/karpathy/minGPT）
読む順番：
1. `README.md`（何ができるか、adder/chargptの違い）
2. `mingpt/model.py`
   - `NewGELU.forward`
   - `CausalSelfAttention.__init__` → `forward`（`c_attn` の `3*` の意味、`split`、`view/transpose`、mask、`softmax`、`dropout`、`proj`）
   - `Block.forward`（pre-LN）
   - `GPT.__init__`（`wte`, `wpe`, `drop`, `blocks`, `ln_f`, `head`、`apply(self._init_weights)`）
   - `GPT.forward`（`pos` の作り方、`loss` 計算）
   - `GPT.generate`
3. `mingpt/trainer.py`（`Trainer` の `_train_epoch`, `run`）
4. `projects/adder/adder.py`（データ生成 → 学習 → 評価の流れ）
5. `projects/chargpt/chargpt.py`

### nanoGPT（https://github.com/karpathy/nanoGPT）
読む順番：
1. `README.md` の "install" と "quick start"
2. `data/shakespeare_char/prepare.py`
3. `config/train_shakespeare_char.py`（**ハイパーパラメータの現実的な値**を覚える）
4. `train.py`
   - `get_batch` → `estimate_loss` → `get_lr` → `configure_optimizers` → `main` の訓練ループ（AMP、grad accumulation、clip、eval、ckpt保存）
5. `model.py`（`LayerNorm`, `CausalSelfAttention`（SDPA版）, `MLP`, `Block`, `GPT`）
6. `sample.py` → `bench.py`
7. `config/train_gpt2.py`（**回さない**。読むだけ：GPT-2 124Mの設定）

### その他
| 教材 | URL | 読む場所 |
|---|---|---|
| Llama 1 | https://arxiv.org/abs/2302.13971 | §2.1, 2.2（RMSNorm, SwiGLU, RoPE） |
| RoPE | https://arxiv.org/abs/2104.09864 | §3.2, §3.3 |
| GQA | https://arxiv.org/abs/2305.13245 | §2, Figure 1-2 |
| modded-nanogpt | https://github.com/KellerJordan/modded-nanogpt | `README.md`（Record履歴）→ `train_gpt.py`（**commit固定**） |
| gpt-fast | https://github.com/meta-pytorch/gpt-fast | `generate.py`（`generate`, `speculative_decode`）, `quantize.py` |
| Attention Is All You Need | https://arxiv.org/abs/1706.03762 | §3.2（Scaled Dot-Product Attention）※**後追いでよい** |
| The Annotated Transformer | http://nlp.seas.harvard.edu/annotated-transformer/ | `attention` 関数、`MultiHeadedAttention` クラス |

---

## 1.5 生成AI活用プロンプト集（Phase 1）

### ✅ 良い質問 vs ❌ 悪い質問

| ❌ 悪い | ✅ 良い |
|---|---|
| 「Attentionの実装を教えて」 | 「nanoGPTの`CausalSelfAttention.forward`の以下の3行（`k,v = kv.split(...)`, `q = q.view(...).transpose(1,2)`, `y = y.transpose(1,2).contiguous().view(...)`）について、各ステップのshapeを表にし、`transpose`の後に`contiguous()`が必要な理由をメモリレイアウトの観点で説明して」 |
| 「KV cacheって何？」 | 「私のKV cache実装（下記）で、`block_size`を超える生成をすると何が起きる？ 具体的にT=300, block_size=256のときのテンソルshapeの遷移を追って、破綻箇所を指摘して」 |
| 「lossが下がりません」 | 「自作GPTのval lossが2.4で停滞。nanoGPTは1.5。差分は(a)weight tyingなし (b)初期化std=0.02固定 (c)lr warmupなし。この3つのうち、どれが最も効くかを"予想されるloss差の大きさ"順に並べ、それぞれの確認実験を1つずつ設計して」 |

### コピペ用プロンプト（Phase 1）

```

【P1-1: 行単位のshape追跡】
次の関数について、1行ずつ「入力shape → 出力shape → その演算の意味」を表にしてください。
推測が不確かな行には ? を付けてください。私はその ? の行だけを重点的に検証します。
関数: nanoGPT/model.py の CausalSelfAttention.forward（コード貼付）

```

```

【P1-2: 差分デバッグ（最重要）】
自作GPTとnanoGPTのloss差を埋めたい。以下の条件で、原因候補を「確認コストが低い順」に10個挙げ、
それぞれ「1回のforwardで判定できるチェック方法（コード1行）」を添えてください。
条件: 同データ・同seed・同step数、自作val loss 2.41 / nanoGPT 1.52。
自作の差分: (a) weight tying なし (b) RoPE未導入 (c) dropout 0.0 (d) AdamWのweight decayを全パラメータに適用。

```

```

【P1-3: 反例生成（Attentionマスク）】
私のcausal mask実装:

```python
bias = torch.tril(torch.ones(T, T)).view(1,1,T,T)
att = att.masked_fill(bias[:,:,:T,:T] == 0, float('-inf'))
```

この実装が壊れるケースを3つ、具体的なTと入力で構成してください。
（ヒント: Tがblock_sizeより小さいとき、cacheと併用したとき、の2つは必ず含めること）

```

```

【P1-4: 数式→コード変換の検算】
RoPEの回転行列による定義（下記数式）を、PyTorchで「forループなし・einsumまたはcomplex型」で実装してください。
その後、私の実装（貼付）と数値が一致するか、max abs diff を出すコードを書いてください。
数式: （貼付）

```

```

【P1-5: 推論コストの見積もり訓練】
以下の条件で、KV cacheのメモリ量と、1トークン生成あたりのFLOPsを計算する式を立て、
具体的な数値を出してください。私の計算（貼付）が合っているか検算もしてください。
条件: L=24層, d=1024, n_head=16, n_kv_head=8, T=2048, bf16, batch=8。

```

---

## 1.6 理解度チェック

**説明できるべき概念（各1分）**
- pre-LN と post-LN の違い、どちらが学習しやすいか
- `c_attn` が `3*n_embd` を一括で出す設計の功罪
- weight tying（`wte` と `lm_head` の共有）の効果
- KV cacheが「なぜAttentionの計算量を減らす」のか（減らすのは**再計算**であり、KVの読み出しは残る）
- `scaled_dot_product_attention` が内部で何を最適化しているか（FlashAttention: HBMアクセス削減）

**失敗時のチェックリスト**
- lossが1.5付近で頭打ち → ①データが十分か ②model容量 ③lr ④dropout
- lossが急にnan → ①lr過大 ②AMPのoverflow（GradScalerの`scale`を確認）③maskに`-inf`を入れてsoftmaxでnan（**`-inf`だらけの行がないか**）
- 生成が同じ文字の繰り返し → ①temperature ②訓練不足 ③weight tyingの不整合
- 速度が出ない → ①`contiguous()`漏れ ②`torch.compile`未適用 ③データローダがボトルネック（`num_workers`）

---

## 1.7 つまずきポイントと対処

| つまずき | 対処 |
|---|---|
| `view` と `reshape` の違いで詰まる | **`transpose`の後は必ず`contiguous()`してから`view`**、とだけ覚える。エラーが出たら`.contiguous()`を挟む |
| Attentionのshapeが追えない | 4次元 `(B, nh, T, hs)` を**紙に大きく書いて、矢印で遷移を描く**。これを3回やる |
| 写経してしまい力がつかない | `model.py`を**閉じて**書き、詰まったら「エラーメッセージだけ」AIに投げる（コードは見ない） |
| 自作と本家のloss差が埋まらない | `torch.manual_seed` 固定＋**同じ初期重みをロード**してforward出力を比較（`allclose`）。これで「モデル」か「訓練」かを切り分ける |
| GPUメモリ不足 | `batch_size` を下げる → `block_size` を下げる → AMP → gradient checkpointing（`torch.utils.checkpoint`） |

## 1.8 このフェーズで「やらなくてよいこと」

- ❌ 分散学習（DDP/FSDP/TP）→ Phase 3
- ❌ MoE、Mamba、Linear Attention などの代替アーキテクチャ → Phase 3で読む
- ❌ 学習率スケジューラの自作研究（cosineで十分）
- ❌ BPE tokenizerの自作（`prepare.py`のchar-levelで十分。Phase 2で扱う）
- ❌ 大規模データでの事前学習 → Phase 2
- ❌ 論文の完全理解（Llama論文は§2だけでよい）
