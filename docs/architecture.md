# Architecture and implementation

## Joint action head

The model represents a whole `H×D` command block with one mode variable. For features `h`:

1. Layer normalization and an MLP produce a context vector.
2. A linear head produces K mode logits.
3. The chosen mode embedding and context vector condition an MLP residual.
4. `prototype[k] + scale*tanh(residual)` gives the action chunk.

The fixed prototypes are buffers. During training, masked nearest-prototype assignment provides the mode label and the residual is conditioned on that label. At deployment and validation, the residual uses the predicted mode. This distinction avoids reporting an oracle-conditioned validation action error.

The bounded residual preserves precision beyond the discrete prototypes. Prototype fitting, residual magnitude and mode occupancy should be inspected jointly: a poor codebook can force large corrections, and a dominant mode can hide a regression-like collapse.

## Pretrained OpenVLA adapter

The adapter loads a configured OpenVLA/Prismatic checkpoint, retains its visual backbones, projector and causal language decoder, and obtains decoder features directly rather than materializing vocabulary logits.

```text
BOS | camera-0 patches | camera-1 patches | remaining instruction | state token | joint query
                                                                                  ↓
                                                                      joint mode/residual head
```

Each camera passes through the original single-image vision tower; projected patches are concatenated. The original visual weights and projector are frozen by default. Learned camera offsets, a projected proprioceptive token, one learned joint query and LoRA adapt the context to direct action decisions. A two-camera model adds visual tokens and has a different input condition from original one-camera OpenVLA.

Text uses the original OpenVLA prompt template and space token. Attention masks exclude batch padding and cumulative positions preserve valid token order. The decoder's causal attention lets the final query read the complete preceding observation. This is a single readout token for a joint action variable, so it does not require independent predictions for each coordinate or a bidirectional action-token mask.

Frozen components do not imply unchanged policy behavior. The query, state projection and head require robot supervision. Pretraining transfers visual-language and action-related features; direct action accuracy must be established after adaptation.

## Objective

```text
L = CE(mode_logits, nearest_mode)
    + lambda_brier * Brier(mode_distribution, nearest_mode)
    + lambda_residual * masked_L1(predicted_chunk, demonstrated_chunk)
    + lambda_teacher * KL(teacher_mode_distribution || student_mode_distribution)
```

The teacher term is optional and defaults to zero. CE/Brier against single-demonstration mode labels learns a behavior-mode distribution. It is not a calibrated execution-success objective. The residual head is initialized to zero output, so initial commands equal the selected prototypes.

## Comparisons

`head_type: regression` uses the same backbone and observation/readout pathway with a continuous MLP and L1 loss. It isolates the output-head representation. It is not OpenVLA-OFT's action-token architecture. Reproduce OpenVLA-OFT separately at the pinned upstream revision for the strong external baseline; preserve observation count, proprioception, chunk execution horizon, training data and actual compute budget in any comparison.

The tiny backbone is an image encoder, byte-level text embedding, state projection and small Transformer. It validates the same head, data and infrastructure. It does not model pretrained OpenVLA capability and cannot establish the proposed method's research effectiveness.

Action discretization with continuous correction has prior art in
[BeT (2022)](https://arxiv.org/abs/2206.11251); residual vector quantization appears in
[VQ-BeT (2024)](https://github.com/jayLEE0301/vq_bet_official). Parallel VLA decoding
also predates the recent Jev implementations. The engineering stack enables
controlled research; the combination of these components alone does not establish
a new CVPR contribution.
