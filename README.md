# BALECO

A two-system (fast + deliberation) approach for directional-consistent
natural language inference, developed for **DiCo-NLI** (Track 1: English).

BALECO couples a standard fine-tuned classifier with a lightweight
deliberation stage that is invoked only when needed:

- **System 1 (fast path)**: a fine-tuned DeBERTa-v3-base classifier
  (`train_deberta.py`), applied independently to each ordering of a phrase
  pair.
- **System 2 (deliberation)**: a cross-attention head that reads both
  orderings of a pair jointly and emits a *single* decision over the
  reversal-compatible label pairs
  {(EQUIV,EQUIV), (FWD,BACK), (BACK,FWD), (NEG,NEG)} — directional
  consistency (SoftCons) is guaranteed by construction for every pair it
  takes.
- **Router**: pairs take the fast path by default and are escalated to
  System 2 only if (a) System 1's two predictions are reversal-incompatible
  (a deterministic, zero-cost check) or (b) a learned gate over predictive
  entropy, reversal divergence KL(p1‖P·p2), and rule-based linguistic risk
  features fires under a deliberation budget.

## Repository layout

| File | Description |
|---|---|
| `train_deberta.py` | System 1: fine-tunes DeBERTa-v3-base on the official DiCo-NLI Track 1 train split (starter-kit defaults). |
| `train_dual.py` | BALECO: frozen System 1 + System 2 deliberation head + router; trains on the train split, evaluates with the official scorer, compares against System 1. |
| `requirements.txt` | Pinned dependencies. |

Generated checkpoints, prediction files, and scores are written to
`outputs/` (not committed).

## Setup

1. **Clone the official task repository** (data, starter kit, and official
   scorer — GPL-3.0). Data link and paper mentioned below.

   ```bash
   git clone https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI.git
   export DICONLI_REPO=/path/to/SemEval-2027-Task-2-DiCo-NLI
   ```

   The environment variable (or the `--task_repo` flag on each script)
   tells BALECO where the data (`final_data/`) and the official scorer
   (`evaluation_functions/`) live. By default the scripts look for the task
   repository as a sibling of this one.

2. **Install dependencies** (Python 3.10+):

   ```bash
   pip install -r requirements.txt
   ```

   The default PyPI `torch` wheel covers CUDA (Linux) and MPS (Apple
   Silicon); for a specific CUDA version install the matching wheel first.

## Reproducing the experiments

Run from this directory, on GPU (CUDA or Apple Silicon MPS):

```bash
# Step 1 — System 1 baseline (DeBERTa-v3-base, starter-kit defaults)
python train_deberta.py

# Step 2 — BALECO (System 2 + router on top of the frozen System 1 checkpoint)
python train_dual.py
```

`train_dual.py --help` lists all hyperparameters and ablation switches
(`--tune_s1` for joint fine-tuning, `--gate_routing` for the CE-consequence
gate, `--route_rule` for routing ablations, `--smoke` for a quick test).
`train_dual.py` prints a final comparison table against the System-1 run and
saves the model to `outputs/models/baleco_track1/`.

### Results (Track 1 dev, official scorer)

| System | Weighted F1 | SoftCons | HardCons |
|---|---|---|---|
| Majority | 0.1287 | 0.0000 | 0.0000 |
| Random | 0.2223 | 0.1372 | 0.0433 |
| DeBERTa-v3-base (System 1) | 0.7841 | 0.8845 | 0.8087 |
| **BALECO** | **0.8106** | **1.0000** | **0.8736** |

Majority/random are computed on the same dev split with the official scorer.

## Implementation notes

- **Label order.** The starter kit's `LabelEncoder` orders classes
  alphabetically: BACKWARD_ENTAILMENT=0, EQUIVALENCE=1, FORWARD_ENTAILMENT=2,
  NEGATIVE_OTHER=3. `train_dual.py` must use this order.
- **Official SoftCons.** A pair is soft-consistent iff the first prediction
  is a reversible label (not NEGATIVE_OTHER) and the second prediction equals
  its deterministic reverse (FWD↔BACK swap; EQUIV fixed).
- **NEGATIVE_OTHER instances** are always unpaired singles; pair gold labels
  span EQUIVALENCE/FORWARD/BACKWARD only.
- Token lengths on Track 1 have p99 ≈ 18, so `train_dual.py` uses
  `max_length=64` (no accuracy cost, large speedup); `train_deberta.py`
  keeps the starter-kit default of 256.

## Citation

The data is released with the below paper.

**Data.** The DiCo-NLI dataset, the starter kit, and the official scorer
come from the [official task repository]
(https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI) and are not
redistributed here. If you use the data, please cite:

> Apaolaza Larraya, Jon Felix, Begoña Altuna, Aitor Soroa, and Inigo
> Lopez-Gazpio. 2026. Assessing Logical Coherence of LLMs via Fine-Grained
> NLI. In *Proceedings of the Fifteenth Language Resources and Evaluation
> Conference (LREC 2026)*, pages 5431–5444, Palma, Mallorca, Spain.
> European Language Resources Association.

**Starter code.** `train_deberta.py` drives the Trainer from the task's
official starter kit and both scripts score predictions with the task's
official `evaluation_functions` package, both from the repository above.

**Pretrained model.** The System-1 backbone is DeBERTa-v3-base:

> He, Pengcheng, Xiaozhong Liu, Jianfeng Gao, and Weizhu Chen. 2023.
> DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with
> Gradient-Displaced Token Replacement. arXiv preprint arXiv:2111.09543.
