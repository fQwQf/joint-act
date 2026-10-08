# Third-party provenance

The JointAct modules are newly written. They adapt public design patterns and preserve the following input/output conventions:

- **OpenVLA / OpenVLA-OFT**: MIT; authors Moo Jin Kim, Chelsea Finn, Percy Liang and collaborators. Reference commit `e4287e94541f459edc4feabc4e181f537cd569a8` of [OpenVLA-OFT](https://github.com/moojink/openvla-oft). Its `libero_dataset_transform`, `ActionTokenizer`, and LIBERO evaluation explain gripper inversion, action token bins, camera orientation, and settling steps. Our implementation follows those conventions with independently written functions. The upstream license is retained in `licenses/openvla-oft-MIT.txt`.
- **Open-Jev**: [repository](https://github.com/Zefan-Cai/Open-Jev), MIT. Read as a reference for pretrained-backbone candidate scoring and LoRA; source is not copied into this package.
- **OneJev**: [repository](https://github.com/OmniJev/OneJev), Apache-2.0. Read as a reference for restricted decision supervision; source is not copied into this package.
- **Pretrained OpenVLA**: downloaded separately from the explicitly configured model repository. Its weights, remote Python model implementation, and tokenizer retain their upstream licenses. A JointAct adapter bundle does not redistribute frozen foundation weights.
- **LIBERO**: [repository](https://github.com/Lifelong-Robot-Learning/LIBERO). Data and simulator assets are obtained separately and retain upstream terms. The converted dataset records its origin and transformation; conversion does not relicense it.
- Optional Python dependencies retain their own licenses. Installation does not imply permission to redistribute model or dataset assets.
- **Behavior Transformer / VQ-BeT**: the broader discrete-mode plus continuous-correction idea predates Jev. See [BeT (2022)](https://arxiv.org/abs/2206.11251) and [VQ-BeT's official implementation (2024)](https://github.com/jayLEE0301/vq_bet_official). These are conceptual prior art; no source from these repositories is vendored. The current joint head is an experimental adaptation, not a claim to have originated that mechanism.

`third_party/` is an ignored reference checkout, not part of the distributable package. `scripts/fetch_upstream.sh` reproduces its pinned contents. No proprietary Jev code, weights or training data are used.

## Supervision implementation references (2026-10-08)

- [miniBET `bet.py`, pinned source](https://github.com/notmahi/miniBET/blob/641df6191228764f5e57e009a394d1d6c9eef28f/behavior_transformer/bet.py#L187-L201): nearest-cluster labels supervise classification and the matching offset; sampled clusters are used for returned predictions.
- [VQ-BeT `bet.py`, pinned source](https://github.com/jayLEE0301/vq_bet_official/blob/09d4851288ca5deaaa1ab367a208e520f8ee9a84/vq_behavior_transformer/bet.py#L252-L295): sampled discrete codes select offsets; the decoded action plus sampled offsets receives L1 supervision. Prediction-conditioned correction is therefore existing practice.
- [OpenVLA-OFT action head](https://github.com/moojink/openvla-oft/blob/e4287e94541f459edc4feabc4e181f537cd569a8/prismatic/models/action_heads.py#L84-L107): its MLPResNet consumes per-action token hidden states. Our single-readout MLP regression control has a different architecture and remains an internal control, not an OFT reproduction.

The new losses are independently implemented. They retain the demonstrated-mode anchor, gate additional argmax-mode correction by proximity and reachability, and optionally minimize expected fixed-prototype action error. These implementation differences are hypotheses to test, not evidence of effectiveness or a claim of literature-wide novelty. No upstream source was copied.
