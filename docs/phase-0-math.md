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
