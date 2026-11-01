# プロレベルLLM開発者への実践ロードマップ教科書

### 〜OSSコード × 生成AI対話で「訓練・改良・デプロイ」を手に馴染ませる28週間〜

---

## この教科書の使い方

### 0. あなた専用にチューニングした前提

| 項目           | あなたの状態                               | 本書の扱い                                                                                   |
| -------------- | ------------------------------------------ | -------------------------------------------------------------------------------------------- |
| 数学           | 線形代数・微積・確率は履修済みだが忘却気味 | **定義から入らず、テンソル形状とコードから入る**。数式は「コードの言い換え」として再導入する |
| プログラミング | Python可、CS基礎あり                       | 言語文法の説明はしない。**設計・形状・数値の話に全振り**                                     |
| 関心           | LLMの訓練・改良・デプロイ                  | API利用・プロンプトエンジニアリング・RAGは**原則スコープ外**（付録に最小限のみ）             |
| 学習スタイル   | OSSを読み、生成AIに都度質問                | 全フェーズを**「読む順番 → 質問 → 書く → 走らせる」**のループで設計                          |
| 計算資源       | 個人〜小規模。H100クラスタなし             | **「回す教材」と「読むだけ教材」を各章で明示的に分離**。回すのは≤1Bモデル中心                |

### 1. この本の3つの約束

**(A) コード → 数式 → 概念 の往復**
抽象論から入らない。「このテンソルのshapeは？」「この1行は何次元目を潰している？」から始め、必要になった瞬間にだけ数式を呼び出す。数式を見たら必ず「PyTorchで1行にすると？」を自問する。

**(B) 生成AIは「答えを教える人」ではなく「思考を深める相手」**
「これ何？」と聞くのは非効率。**制約を課す質問**（「FlashAttentionを使わずに書いて」）、**反例を求める質問**（「この実装が壊れる入力は？」）、**形状を追わせる質問**（「(B,T,C)がこの行を通るとどう変わる？1行ずつ表にせよ」）を使う。各章にプロンプト例を置く。

**(C) 「訓練できる」はプロの入口に過ぎない**
プロの条件は **「推論コストを制御できる」** こと。Phase 1の最終週からKV cacheとスループット計測を常に意識させる。

### 2. 進め方のルール（挫折防止）

1. **1日1コミット**：コードでもメモでもよい。学習ログ（付録C）に必ず残す。
2. **15分ルール**：15分詰まったら生成AIに質問する。ただし**「自分の仮説」を必ずプロンプトに含める**。
3. **週末は「説明会」**：その週の概念を、生成AI相手に5分で説明する（AIに「素人向けに説明した内容の誤りを指摘して」と頼む）。
4. **完璧主義の禁止**：各章の「やらなくてよいこと」は**本当にやらない**。スコープを守ることが最大の生産性。
5. **計算資源は「時間単価」で考える**：クラウドGPUはスポット/オンデマンドを使い分け、**1実験あたりの上限額を先に決める**（例：1実験 $5）。

### 3. 想定学習時間

- 標準：**週12時間**（平日2h×4 + 週末4h）→ 約28週でPhase 4到達
- 短縮版：週20時間 → 約18週
- 各週の「想定時間」は標準ペースで記載。**足りない週は翌週に持ち越さず、「わかったこと/わからないこと」をログに残して前進する**（LLM開発は反復が本質）。

---

## 全体マップ

| Phase    | 期間                | テーマ                                | 主な成果物（＝検証可能なアウトプット）                                | 計算資源                               |
| -------- | ------------------- | ------------------------------------- | --------------------------------------------------------------------- | -------------------------------------- |
| **0**    | 3週 (36h)           | 数学の実践的最小セット再構築          | micrograd相当の自動微分エンジン自作、CE/AdamWの手実装、形状チェック表 | CPU可（GPU不要）                       |
| **1**    | 5週 (60h)           | Transformer核心理解（minGPT→nanoGPT） | 300行GPTの自作＋char-level訓練、KV cache実装とスループット計測        | 単一GPU 8〜16GB or Colab               |
| **2**    | 6週 (72h)           | 完全オープンOSSでの再現体験           | 自作データパイプライン＋100M級モデルの事前学習＋評価レポート          | 単一GPU 16〜24GB or スポットA100数時間 |
| **3**    | 5週 (60h)           | プロダクションレシピの読解            | Nemotron/Llama3/DeepSeek/OLMo2の比較チェックリスト、小規模移植計画    | 基本読むだけ（一部小実験）             |
| **4**    | 8週+ (96h+)         | 独自実験とデプロイ                    | SFT/LoRA/DPO/GRPOの実施、量子化、vLLMデプロイ、コスト試算表、公開     | 単一GPU 24GB or 従量クラウド           |
| **合計** | **約27週（+予備）** |                                       | **公開可能な小規模LLMと技術レポート**                                 | 総GPU費用目安 **$100〜400**            |

> **費用目安の根拠**：Colab Pro（T4/L4）＋スポットA100 40GB（$1〜1.5/h）× 数十時間＋RTX 4090 overnight 数回。Phase 2の100M級事前学習は「A100 8〜12時間」で収まる規模に意図的に抑えている。

---

# Phase 0：数学の実践的最小セットの再構築（3週間）

## 0.1 フェーズの目的と到達基準

**目的**：数学を「試験科目」ではなく「デバッグ道具」として再起動する。具体的には、**テンソルの形状と数式を相互翻訳できる**状態を作る。ここが一番挫折しやすいが、ここを飛ばすとPhase 1以降で「lossが下がらない理由がわからない」地獄に入る。

**到達基準（すべて口頭/ホワイトボードで説明できたら次へ）**

1. `micrograd` の `Value` クラス相当を、**何も見ずに** 150行以内で書き直せる（forward/backward/topological sort）。
2. 任意のPyTorchコードを見て、**全中間テンソルのshapeを紙に書ける**（broadcasting規則含む）。
3. `cross_entropy(logits, targets)` を **`logsumexp` を使って手書き**でき、なぜ数値安定化が必要か説明できる。
4. AdamWの更新式（momentum、bias correction、decoupled weight decay）を**コードで書ける**。
5. 「学習率を上げたらlossが発散する」現象を、**勾配ノルムと更新幅の言葉で**説明できる。

**移行条件**：上記1〜5を、生成AIに採点させて80点以上（プロンプトは後述）。

---

## 0.2 週単位スケジュール

| 週      | テーマ                             | 想定時間 | 主教材                                                                                                       |
| ------- | ---------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------ |
| **0-1** | 自動微分を自作する（micrograd）    | 12h      | `karpathy/micrograd`、Karpathy "The spelled-out intro to neural networks and backpropagation"                |
| **0-2** | 線形代数・確率をテンソル演算として | 12h      | PyTorch公式チュートリアル、`einsum`、CS231n backprop notes、`The Matrix Calculus You Need For Deep Learning` |
| **0-3** | 最適化・数値安定性・初期化         | 12h      | nanoGPT `model.py` の該当箇所、AdamW論文(1711.05101)、Mixed Precision Training(1710.03740)                   |

---

## 0.3 日単位タスク例

### Week 0-1：自動微分を自作する

| Day      | タスク（読む／書く／走らせる／質問する）                                                                                                                                                                                     |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Day1** | 【環境】WSL2 or Linux、Python 3.11、`uv` or conda、PyTorch(CUDA版)、`nvidia-smi`確認、`torch.cuda.is_available()`。**走らせる**：`torch.randn(2,3) @ torch.randn(3,4)` のshape確認。学習ログ雛形作成（付録C）                |
| **Day2** | 【読む】`micrograd/micrograd/engine.py` を上から。`Value.__init__`, `__add__`, `__mul__`, `__pow__`, `tanh`, `backward`, `_backward` クロージャ。【質問】「なぜ `_backward` をクロージャで持つのか、関数ポインタではダメか」 |
| **Day3** | 【読む】`engine.py` の `backward()` 内トポロジカルソート。【書く】`Value` を**写経せず**自分の言葉で再実装（まず加算・乗算のみ）。【走らせる】`f = (a*b + c).tanh()` で手計算と一致するか                                    |
| **Day4** | 【読む】`micrograd/nn.py` の `Neuron`, `Layer`, `MLP`。【書く】`__pow__`, `__truediv__`, `relu` を追加。【走らせる】小さな回帰タスクでlossが下がるか                                                                         |
| **Day5** | 【書く】**勾配チェック**：`torch.autograd` と自作エンジンの勾配を数値比較（`torch.autograd.gradcheck`相当を自作）。【質問】「自作実装が合わないとき、どのノードから疑うべきか優先順位を教えて」                              |
| **週末** | 【説明会】micrograd全体を図解（計算グラフ＋逆伝播）。生成AIに「この説明の誤りを3点指摘して」と投げる                                                                                                                         |

**実装課題 0-A**：`micrograd`互換の `Value` を `mygrad/engine.py` に自作し、`mygrad/nn.py` で2層MLPを作り、`sklearn.datasets.make_moons` を分類せよ（精度85%以上）。**numpyのみ許可、PyTorch autograd禁止**。

---

### Week 0-2：線形代数・確率をテンソル演算として

| Day      | タスク                                                                                                                                                                                                                  |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Day1** | 【読む】PyTorch "Tensors" チュートリアル。【書く】`(B,T,C)` の3次元テンソルに対し、`view/reshape/permute/transpose/contiguous` を全部試し、**shapeの変化を表に記録**                                                    |
| **Day2** | 【読む】Broadcasting規則（PyTorch docs "Broadcasting semantics"）。【走らせる】`(4,1,8) + (1,5,8)` → ? を予測してから実行。**予測と実行のズレをログに残す**                                                             |
| **Day3** | 【読む】`torch.einsum` ドキュメント。【書く】Attentionの `Q@K^T`、`softmax@V`、batch matmulを**einsumで書き直す**。`einsum('bthd,bshd->bhts', q,k)` の意味を日本語で説明                                                |
| **Day4** | 【読む】CS231n "Backpropagation" notes。【書く】softmax + cross-entropy を、`logsumexp` トリック込みで**手書き**。`F.cross_entropy` と数値一致を確認。【質問】「logitsが1000のとき素朴な実装はなぜinfになる？具体値で」 |
| **Day5** | 【読む】最尤推定とCEの関係（任意の記事でよいが、必ず「なぜCE＝負の対数尤度か」を導出）。【書く】temperature / top-k / top-p サンプリングを10行で実装                                                                    |
| **週末** | 【書く】「shapeチェック表」テンプレート作成（下記）＋説明会                                                                                                                                                             |

**shapeチェック表テンプレート（以後全フェーズで使う）**

```
| 変数 | 意味 | shape | dtype | 備考 |
| x    | 入力token | (B,T) | int64 | |
| h    | 埋め込み後 | (B,T,C) | bf16 | |
```

---

### Week 0-3：最適化・数値安定性・初期化

| Day      | タスク                                                                                                                                                                                                                    |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Day1** | 【読む】SGD → Momentum → Adam（原論文 or 良記事）。【書く】SGD/momentum/Adam/AdamWを各20行で実装し、2次関数 `f(x)=(x-3)^2` で収束比較                                                                                     |
| **Day2** | 【読む】AdamW論文(arXiv:1711.05101)の**Section 2のみ**（decoupled weight decay）。【質問】「AdamにL2正則化を入れるのとAdamWの違いを、更新式レベルで説明して」                                                             |
| **Day3** | 【読む】nanoGPT `model.py` の `GPT.configure_optimizers`（decay/no-decayの振り分け）。【書く】同じ振り分けを自作                                                                                                          |
| **Day4** | 【読む】Mixed Precision Training(arXiv:1710.03740) のloss scaling部分。【走らせる】fp16/bf16/fp32で同じ小モデルを訓練し、**loss曲線とメモリ使用量**を比較                                                                 |
| **Day5** | 【読む】初期化（GPT-2の `std=0.02`、`_init_weights`、residual scaling `1/sqrt(2N)`）。【書く】初期化を変えて深いMLP（20層）の活性値分散を測定。【質問】「残差ストリームの分散が層を追うごとに爆発する条件を、式で示して」 |
| **週末** | 【書く】学習率スケジュール（warmup + cosine）を実装し、warmup有無で挙動比較。**Phase 0 修了テスト**（下記）                                                                                                               |

---

## 0.4 使用OSS・教材と「どこを読むか」

| 教材                    | URL                                                     | 読む場所（ファイル/関数レベル）                                                                                                                           |
| ----------------------- | ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| micrograd               | https://github.com/karpathy/micrograd                   | `micrograd/engine.py`（全行・全関数）、`micrograd/nn.py`（`Neuron.forward`, `Layer`, `MLP`）、`test/test_engine.py`（**テストを先に読むと仕様がわかる**） |
| Karpathy "Zero to Hero" | https://karpathy.ai/zero-to-hero.html                   | 第1講 "micrograd"、第2講 "makemore part1"（bigram→NN）                                                                                                    |
| makemore                | https://github.com/karpathy/makemore                    | `makemore/lecture1_bigram.ipynb` 〜 `lecture3_mlp.ipynb` の**損失計算セルだけ**拾い読み                                                                   |
| PyTorch docs            | https://pytorch.org/docs/stable/notes/broadcasting.html | Broadcasting semantics 全体                                                                                                                               |
| CS231n notes            | https://cs231n.github.io/optimization-2/                | Backpropagation 全体                                                                                                                                      |
| Matrix Calculus         | https://explained.ai/matrix-calculus/                   | §1〜§5（ベクトル→ベクトルの微分、Jacobian）                                                                                                               |
| AdamW                   | https://arxiv.org/abs/1711.05101                        | Section 2（Decoupled Weight Decay）のみ                                                                                                                   |
| Mixed Precision         | https://arxiv.org/abs/1710.03740                        | Section 3（Loss Scaling）のみ                                                                                                                             |
| nanoGPT（先読み）       | https://github.com/karpathy/nanoGPT                     | `model.py` の `configure_optimizers` だけ先に読む（Phase 1で全部読む）                                                                                    |

---

## 0.5 生成AI活用プロンプト集（Phase 0）

### ✅ 良い質問 vs ❌ 悪い質問

| ❌ 悪い質問                        | ✅ 良い質問                                                                                                                                                                                                                        |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 「バックプロパゲーションを教えて」 | 「以下の自作Value.backward()のコードで、`_backward`をトポロジカル順に呼ぶ必要がある理由を、具体的にどのノードで破綻するかを例示して説明して。私の理解は『共有ノードの勾配が上書きされるから』だが合っている？ コード: `（貼付）`」 |
| 「AdamWって何？」                  | 「AdamWの更新式で、weight_decayを勾配に足す実装とパラメータを直接減衰させる実装を両方書いて、`lr=1e-3, wd=0.1` のとき1ステップでパラメータが何%変わるか数値で比較して」                                                            |
| 「shapeが合いません」              | 「`x: (4, 8, 16)`, `w: (16, 32)` に対して `x @ w` がエラーになる理由と、正しいshapeの合わせ方を3通り（reshape/permute/einsum）示して。それぞれのメモリ上の連続性の違いも教えて」                                                   |

### コピペ用プロンプト（Phase 0）

````
【P0-1: コード→数式の翻訳】
次のPyTorchコードを、数式（行列記法）とテンソルshapeの遷移表（各行ごとに shape と dtype）に翻訳してください。
特に broadcasting が発生する行には ★ を付け、なぜその shape で成立するか1行で説明してください。
コード:
```python
（貼付）
````

```

```

【P0-2: 反例を出させる】
私の実装した cross_entropy（下記）が数値的に壊れる入力を3つ、具体的な数値で構成してください。
それぞれ「どの行で何が起きるか（inf/nan/overflow）」と「logsumexp を使った修正版」を示してください。
コード:

```python
（貼付）
```

```

```

【P0-3: 形状を追わせる（検算モード）】
次のコードの各テンソルのshapeを、あなたが推測した上で表にしてください。
その後、私の予測表（下記）と比較し、食い違いを指摘してください。私の予測:
（表を貼付）
コード:

```python
（貼付）
```

```

```

【P0-4: 採点者として使う（修了テスト）】
あなたはLLM開発の面接官です。以下の5問を私に出題し、私の回答を採点してください。
基準: 80点未満なら不合格。間違いは「なぜ間違いか」をコードレベルで指摘。
出題範囲: (1)自作autogradのbackward順序 (2)broadcasting (3)logsumexp付きCE (4)AdamW更新式 (5)学習率过大での発散理由。
1問ずつ出題し、私の回答を待ってください。

```

```

【P0-5: つまずきの言語化】
私は「AdamWのbias correctionがなぜ必要か」が腹落ちしていません。
step=1, beta1=0.9, grad=1.0 のとき、bias correctionあり/なしの m_hat を数値で示し、
「学習初期に更新が過小になる」ことを定量的に説明してください。

```

---

## 0.6 理解度チェック

**自己クイズ（答えられるか）**
1. `y = x @ W + b`（x:(B,T,C), W:(C,D)）で、bのshapeは？ なぜ `(D,)` で放送できる？
2. `softmax` の分母に `max` を引くのはなぜ？ 引かないとどんな入力で壊れる？
3. `loss.backward()` の前に `optimizer.zero_grad()` を忘れると何が起きる？
4. `torch.float16` でlossが `nan` になった。疑う順番は？（loss scaling → softmax → layernorm → 勾配爆発）
5. 深いネットワークで残差接続がない場合、なぜ勾配が消える？ 数式で。

**失敗時のチェックリスト**
- 勾配が合わない → ①数値勾配のeps（1e-4）②`requires_grad` ③in-place演算 ④dtype（fp16でgradcheckは不可）
- lossが動かない → ①lr ②ラベルとlogitsの向き ③`zero_grad` ④データがシャッフルされているか
- nan → ①log(0) ②expオーバーフロー ③0除算 ④初期化

---

## 0.7 よくあるつまずきと対処

| つまずき | 対処 |
|---|---|
| 「数学の教科書を最初から読み直したい」衝動 | **禁止**。必要になった定義だけ、コードの直後に引く。教科書は索引として使う |
| microgradの `_backward` クロージャが読めない | Pythonのクロージャを10分だけ復習 → 「`out._backward = lambda: ...` が呼ばれる瞬間」をprintデバッグで確認 |
| shapeが頭に入らない | **全ての関数に `assert x.shape == (...)` を書く**。実行時アサートが最強の教材 |
| einsumがわからない | まず `torch.einsum('ij,jk->ik', A, B)` = `A@B` を確認。**添字を消す＝その軸を総和**と覚える |
| GPUがなくて不安 | Phase 0は**CPUで完結**する設計。GPUはPhase 1から |

## 0.8 このフェーズで「やらなくてよいこと」

- ❌ 測度論、ルベーグ積分、確率論の公理系
- ❌ 行列のLU分解・固有値分解の手計算（概念だけ知っていればよい）
- ❌ 多変数のテイラー展開の厳密な証明
- ❌ numpyの網羅的API暗記
- ❌ 論文の英語全文読解（Abstract + 該当Sectionのみ）
- ❌ 自作autogradの高速化（Phase 0では**正しさだけ**を追う）

---

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

---

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

---

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

---

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

---

# フェーズ移行条件まとめ

| 移行 | 条件（すべて満たす） |
|---|---|
| **0 → 1** | 自作autogradが動く / shapeチェック表が書ける / 手書きCEが`F.cross_entropy`と一致 / AdamW更新式をコードで書ける / 修了テスト80点 |
| **1 → 2** | 自作GPTが300行で書ける / shakespeare_charでval loss 1.5台 / KV cache自作＋レイテンシグラフ / モダン部品4つを説明できる |
| **2 → 3** | 100Mモデルの事前学習を完遂 / ドメイン別perplexityを出せる / OLMoとnanoGPTの差分10項目 / レポート公開 |
| **3 → 4** | 5軸比較表の完成 / 小規模移植を3つ実施 / GRPOの目的関数をコードに落とせる / 移植計画書 |
| **4 → 継続** | SFT/DPO or GRPOのbefore-after / 量子化トレードオフ表 / vLLMデプロイ＋コスト試算 / 公開完了 / 次の問い3つ |

---

# 付録A：用語集

| 用語 | 一言定義 | コード上の現れ |
|---|---|---|
| **logits** | softmax前の生スコア | `lm_head(h)` の出力 `(B,T,V)` |
| **cross-entropy** | 正解トークンの負の対数尤度の平均 | `F.cross_entropy(logits.view(-1,V), targets.view(-1))` |
| **perplexity** | `exp(loss)`。モデルの「迷い具合」 | `math.exp(val_loss)` |
| **causal mask** | 未来を見ないための上三角マスク | `torch.tril(...)` / `is_causal=True` |
| **pre-LN / post-LN** | LayerNormをAttentionの前/後に置く | `x = x + attn(ln(x))` がpre-LN |
| **RMSNorm** | 平均を引かずRMSで正規化 | `x * torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+eps) * g` |
| **RoPE** | 位置を回転で埋め込む | `apply_rotary_emb(q, cos, sin)` |
| **GQA** | KVヘッドを共有してKV cache削減 | `n_kv_head < n_head` |
| **MLA** | 低ランク潜在表現でKVを圧縮 | DeepSeek-V3 |
| **KV cache** | 過去のK/Vを保持して再計算を防ぐ | `past_key_value` |
| **PagedAttention** | KV cacheをページ単位で管理 | vLLM |
| **FlashAttention** | HBMアクセスを減らす厳密Attention | `F.scaled_dot_product_attention` |
| **weight tying** | 入力埋め込みと出力射影を共有 | `self.lm_head.weight = self.wte.weight` |
| **gradient accumulation** | 小バッチを溜めて大バッチ相当に | `loss = loss / accum; loss.backward()` |
| **AMP** | 混合精度学習 | `torch.amp.autocast`, `GradScaler` |
| **z-loss** | logsumexpの二乗を罰する安定化項 | OLMo 2 |
| **QK-norm** | Q,Kを正規化してlogit爆発を防ぐ | OLMo 2 |
| **LoRA** | 低ランク差分だけ学習 | `B @ A * (alpha/r)` |
| **QLoRA** | 4bit量子化＋LoRA | `bitsandbytes` |
| **DPO** | 選好データから直接方策を最適化 | `-logsigmoid(β(Δchosen - Δrejected))` |
| **GRPO** | group内正規化のpolicy gradient | `advantage = (r - mean(r)) / std(r)` |
| **RLVR** | 検証可能な報酬によるRL | 答え合わせ・テスト通過 |
| **TTFT / TPOT** | 初回トークンまでの時間 / 1トークンあたり時間 | vLLM benchmark |
| **continuous batching** | リクエストを動的にバッチ化 | vLLM scheduler |

---

# 付録B：推奨リポジトリ一覧（ブックマーク用）

### 学習・自作系
| リポジトリ | URL | 用途 |
|---|---|---|
| karpathy/micrograd | https://github.com/karpathy/micrograd | Phase 0 |
| karpathy/makemore | https://github.com/karpathy/makemore | Phase 0 |
| karpathy/minGPT | https://github.com/karpathy/minGPT | Phase 1 |
| karpathy/nanoGPT | https://github.com/karpathy/nanoGPT | Phase 1 |
| karpathy/nanochat | https://github.com/karpathy/nanochat | Phase 2-4（**要確認**：フルパイプラインの構成は変更が早い） |
| KellerJordan/modded-nanogpt | https://github.com/KellerJordan/modded-nanogpt | Phase 1（**commit固定**） |
| meta-pytorch/gpt-fast | https://github.com/meta-pytorch/gpt-fast | Phase 1, 4 |
| Stanford CS336 | https://stanford-cs336.github.io/ | 全Phase（講義資料＋assignment） |

### オープンLLM再現系
| リポジトリ | URL | 用途 |
|---|---|---|
| allenai/OLMo | https://github.com/allenai/OLMo | Phase 2, 3 |
| allenai/OLMo-core | https://github.com/allenai/OLMo-core | Phase 2, 3（要確認） |
| allenai/dolma | https://github.com/allenai/dolma | Phase 2 |
| huggingface/smollm | https://github.com/huggingface/smollm | Phase 2 |
| huggingface/nanotron | https://github.com/huggingface/nanotron | Phase 3 |
| NVIDIA/Megatron-LM | https://github.com/NVIDIA/Megatron-LM | Phase 3 |
| NVIDIA/NeMo | https://github.com/NVIDIA/NeMo | Phase 3 |
| pytorch/torchtitan | https://github.com/pytorch/torchtitan | Phase 3 |
| Lightning-AI/litgpt | https://github.com/Lightning-AI/litgpt | Phase 2, 3 |

### 微調整・選好学習
| リポジトリ | URL | 用途 |
|---|---|---|
| huggingface/trl | https://github.com/huggingface/trl | Phase 4 |
| huggingface/peft | https://github.com/huggingface/peft | Phase 4 |
| unslothai/unsloth | https://github.com/unslothai/unsloth | Phase 4 |
| axolotl-ai-cloud/axolotl | https://github.com/axolotl-ai-cloud/axolotl | Phase 4 |

### 推論・デプロイ
| リポジトリ | URL | 用途 |
|---|---|---|
| vllm-project/vllm | https://github.com/vllm-project/vllm | Phase 4 |
| sgl-project/sglang | https://github.com/sgl-project/sglang | Phase 4（vLLMの代替） |
| ggml-org/llama.cpp | https://github.com/ggml-org/llama.cpp | Phase 4 |
| vllm-project/llm-compressor | https://github.com/vllm-project/llm-compressor | Phase 4 |
| casper-hansen/AutoAWQ | https://github.com/casper-hansen/AutoAWQ | Phase 4 |
| EleutherAI/lm-evaluation-harness | https://github.com/EleutherAI/lm-evaluation-harness | Phase 2, 4 |

### データ
| リソース | URL |
|---|---|
| FineWeb-Edu | https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu |
| Cosmopedia | https://huggingface.co/datasets/HuggingFaceTB/cosmopedia |
| Dolma | https://huggingface.co/datasets/allenai/dolma |
| TinyStories | https://huggingface.co/datasets/roneneldan/TinyStories |

---

# 付録C：学習ログのつけ方

### C.1 なぜログか
LLM開発は**「同じ失敗を3回やる」**のが最大のコスト。ログは「未来の自分への手紙」であり、Phase 4の「次の問い」の源泉であり、ポートフォリオでもある。

### C.2 テンプレート（1日1エントリ、5分で書く）

```markdown
## YYYY-MM-DD / Phase X-Y / Day N

### 今日やったこと（1行）
nanoGPTのCausalSelfAttentionをeinsumで書き直した

### 読んだコード・論文（ファイル/関数/章）
nanoGPT/model.py: CausalSelfAttention.forward L40-60

### わかったこと（自分の言葉で）
- `(B,T,C)` を `(B,nh,T,hs)` にするとき、`view` は最後の次元を分割する。
- einsumだと `bthd,bshd->bhts` でマスク前のスコアが出る。

### わからなかったこと（→ 翌日の問い）
- `contiguous()` が不要になる条件は？

### 生成AIに聞いたこと（プロンプトと要点）
- P1-1を使ってshape表を作成。`transpose`後の`view`でエラーになる理由が判明。

### 数値（実験した場合）
val_loss=1.52, lr=1e-3, throughput=42k tokens/s, GPU mem=11.2GB

### 明日やること（1つだけ）
KV cacheの実装に着手
````

### C.3 週次レビュー（週末30分）

1. **今週の「わかった」を5つ**、人に説明できる粒度で書く。
2. **未解決の問い**を3つ選び、優先順位をつける。
3. **来週の計画**を「Day1〜5」で書く（本教科書の該当週と照合）。
4. **コスト**（GPU時間 × 単価）を記録。累計を出す。
5. **生成AIへの質問ログ**を見返し、「良い質問だった」ものを3つ選んで再利用可能な形で保存。

### C.4 問いの種リスト（Phase 4への橋渡し）

各Phaseで必ず「未解決の問い」をストックする。例：

| Phase | 問いの種の例                             | Phase 4での実験            |
| ----- | ---------------------------------------- | -------------------------- |
| 0     | なぜAdamWはAdamより安定するのか          | 同一条件で比較             |
| 1     | KV cacheのメモリを減らすには？           | GQA/MLAの効果測定          |
| 1     | 学習率スケジュールはlossにどれだけ効くか | warmup/constant/cosine比較 |
| 2     | データのドメイン混合比はどう決めるか     | 混合比ablation             |
| 2     | QK-normは本当に効くか                    | 有無比較                   |
| 3     | GRPOのgroup sizeはどう効くか             | G=2/4/8/16比較             |
| 3     | 量子化で劣化しやすいタスクは？           | タスク別劣化測定           |

### C.5 公開用チェックリスト（Phase 4 完了時）

- [ ] README に「環境構築 / 再現コマンド / 期待される出力」がある
- [ ] `configs/` に全実験の設定が残っている（**seed固定**）
- [ ] W&B / TensorBoard のログが公開またはスクショで残っている
- [ ] モデルカードに「用途・制限・訓練データ・評価・倫理」がある
- [ ] コスト試算表（学習・推論）がある
- [ ] 失敗した実験と、その理由が書かれている（**これが一番価値がある**）
- [ ] 次の問いが3つ、実験計画付きで書かれている

---

## 最後に：この教科書の使い方（再掲・最重要）

1. **各週の冒頭で「到達基準」を読み、週末に自己採点する。**
2. **コードを閉じて書く時間を必ず作る**（写経は学習ではない）。
3. **生成AIには「仮説」と「コード」と「ログ」をセットで渡す。** これが質問の質を決める。
4. **「やらなくてよいこと」を守る。** スコープを守ることが、28週間走り切る唯一の方法。
5. **詰まったら、15分でAIに聞く。ただし答えをコピペせず、「なぜそうなるか」を自分の言葉でログに書いてから先へ進む。**

> 「訓練できる」は入口。「推論コストを制御できる」がプロの条件。「自分の問いを立てて実験できる」がプロフェッショナル。この28週間は、その3つを順に手に入れる旅です。
