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
