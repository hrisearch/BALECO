#!/usr/bin/env python3
"""BALECO: two-system (fast + deliberation) model for DiCo-NLI Track 1.

Architecture:
  - Shared DeBERTa-v3 backbone, weight-shared over both orderings of a phrase
    pair, initialized from the trained System-1 checkpoint.
  - System 1: the checkpoint's classification head, per direction (frozen by
    default; --tune_s1 unfreezes it for joint fine-tuning).
  - Pair signals: predictive entropy, reversal divergence KL(p1 || P.p2) with
    P the task's label-reversal permutation, 6 rule-based linguistic risk
    features, and pooled-representation asymmetry.
  - Router: small MLP over the signals -> gate g in (0,1), fast by default
    (final bias init -2), held under a deliberation budget.
  - System 2: bidirectional cross-attention deliberation head over both
    orderings' token sequences -> one 4-way choice over the reversal-compatible
    label pairs {(EQUIV,EQUIV),(FWD,BACK),(BACK,FWD),(NEG,NEG)}; SoftCons = 1
    by construction for deliberated pairs.
  - Losses: gate loss (supervised on System-1 needs-fix pairs, or routing CE
    with --gate_routing), unweighted System-2 pair CE, and a gate budget
    penalty; --tune_s1 adds the reversal-equivariance regularizer.

Trains on the official Track 1 train split, evaluates on the dev split with
the official scorer, and prints a comparison against the System-1 baseline.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
import transformers

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "outputs"


def default_task_repo() -> str:
    """Path to a checkout of the official task repository (data + scorer)."""
    env = os.environ.get("DICONLI_REPO")
    if env:
        return env
    return str(SCRIPT_DIR.parent / "SemEval-2027-Task-2-DiCo-NLI")


# Label order matches the starter kit's LabelEncoder (alphabetically sorted
# classes_): BACKWARD_ENTAILMENT=0, EQUIVALENCE=1, FORWARD_ENTAILMENT=2,
# NEGATIVE_OTHER=3.
LABELS = ["BACKWARD_ENTAILMENT", "EQUIVALENCE", "FORWARD_ENTAILMENT", "NEGATIVE_OTHER"]
L2I = {name: i for i, name in enumerate(LABELS)}
# Reversal permutation P: FWD <-> BACK swap (indices 2<->0), EQUIV and NEG fixed.
PERM = [2, 1, 0, 3]
# Compatible joint label pairs: combo -> (label_original, label_flipped).
COMBOS = [(1, 1), (2, 0), (0, 2), (3, 3)]  # (EQUIV,EQUIV) (FWD,BACK) (BACK,FWD) (NEG,NEG)
# Gold combo for a pair is determined by the original instance's label.
LABEL_TO_COMBO = [2, 0, 1, 3]  # label index -> combo index

QUANTIFIERS = {"all", "some", "many", "most", "no", "none", "few", "every", "each", "several"}
COMPARATIVE_RE = re.compile(r"\b\w+er\b|\b(?:more|less|fewer)\b", re.IGNORECASE)
DETERMINERS = {"a", "an", "the", "this", "that", "these", "those", "his", "her", "their", "its", "my", "your"}
NEGATION_RE = re.compile(r"\b(?:not|never|without|cannot)\b|n't", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_repo", type=str, default=default_task_repo(),
                        help="official DiCo-NLI repository (data, scorer); also settable "
                             "via the DICONLI_REPO environment variable")
    parser.add_argument("--model_path", type=str, default=None,
                        help="System-1 checkpoint; default: outputs/models/deberta_v3_base_track1")
    parser.add_argument("--train_path", type=str, default=None,
                        help="default: <task_repo>/final_data/train/dico_nli_train_track1_participant_labeled.csv")
    parser.add_argument("--dev_path", type=str, default=None,
                        help="default: <task_repo>/final_data/dev/dico_nli_dev_track1_participant_labeled.csv")
    parser.add_argument("--reference_path", type=str, default=None,
                        help="default: <task_repo>/final_data/dev/dico_nli_dev_track1_reference.csv")
    parser.add_argument("--baseline_scores", type=str, default=None,
                        help="System-1 scores.json for the final comparison; default: "
                             "newest outputs/results/deberta_v3_base_track1/predictions_*_official_scores/scores.json")
    parser.add_argument("--save_path", type=str, default=None,
                        help="default: outputs/models/baleco_track1")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="default: outputs/results/baleco_track1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--batch_pairs", type=int, default=32)
    parser.add_argument("--lr_backbone", type=float, default=2e-5)
    parser.add_argument("--lr_heads", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--warmup_pcrt", type=float, default=0.01)
    parser.add_argument("--lambda_rev", type=float, default=0.5)
    parser.add_argument("--gate_budget", type=float, default=0.3)
    parser.add_argument("--lambda_gate", type=float, default=1.0)
    parser.add_argument("--lambda2", type=float, default=1.0, help="weight of the unweighted S2 pair CE (learns S2 even when the gate is closed)")
    parser.add_argument("--tune_s1", action="store_true", help="also fine-tune System 1 (default: frozen, as the trained fast path)")
    parser.add_argument("--gate_routing", action="store_true",
                        help="train the gate by CE consequences (g*CE_S2 + (1-g)*CE_S1); default: supervised gate on S1's mistakes (needs-fix pairs)")
    parser.add_argument("--route_rule", type=str, default="compat+gate", choices=["compat", "gate", "compat+gate"],
                        help="inference routing: deterministic compatibility check, learned gate, or both (default)")
    parser.add_argument("--gate_threshold", type=float, default=0.5)
    parser.add_argument("--smoke", action="store_true", help="tiny subset, 1 epoch, no saving/scoring")
    return parser.parse_args()


def set_seed(seed: int) -> torch.Generator:
    transformers.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tokens(text: str) -> list:
    return re.findall(r"[\w']+|[.,!?;:]", text.lower())


def linguistic_features(t1: str, t2: str) -> list:
    """6 rule-based risk features for the router (control, not content)."""
    w1, w2 = tokens(t1), tokens(t2)
    s1, s2 = set(w1), set(w2)
    q1, q2 = (s1 & QUANTIFIERS), (s2 & QUANTIFIERS)
    quant_asym = float(bool(q1) != bool(q2))
    quant_diff = float(bool(q1) and bool(q2) and (q1 != q2))
    c1, c2 = bool(COMPARATIVE_RE.search(t1)), bool(COMPARATIVE_RE.search(t2))
    comp_asym = float(c1 != c2)
    d1 = w1[0] if w1 else ""
    d2 = w2[0] if w2 else ""
    det_asym = float((d1 in DETERMINERS or d2 in DETERMINERS) and d1 != d2)
    n1, n2 = bool(NEGATION_RE.search(t1)), bool(NEGATION_RE.search(t2))
    neg_asym = float(n1 != n2)
    len_ratio = abs(np.log((len(w1) + 1) / (len(w2) + 1)))
    return [quant_asym, quant_diff, comp_asym, det_asym, neg_asym, len_ratio]


class Baleco(nn.Module):
    """Shared backbone + System 1 head (from checkpoint) + System 2 deliberation + router."""

    def __init__(self, model_path: str, gate_bias_init: float = -2.0):
        super().__init__()
        # System 1 = the trained checkpoint, kept intact (backbone + pooler + head).
        self.s1 = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)
        hidden = self.s1.config.hidden_size  # 768
        # System 2: project to 256, bidirectional cross-attention, pair classifier.
        self.proj = nn.Linear(hidden, 256)
        self.cross_attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        self.pair_fc = nn.Sequential(nn.Linear(512, 256), nn.Tanh(), nn.Linear(256, 4))
        # Router: [H, D, 6 linguistic, h_diff] -> gate.
        self.router = nn.Sequential(nn.Linear(9, 32), nn.ReLU(), nn.Linear(32, 1))
        nn.init.constant_(self.router[-1].bias, gate_bias_init)

    def encode(self, input_ids, attention_mask, token_type_ids):
        out = self.s1(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        return out.logits, out.hidden_states[-1]

    def forward_s1(self, input_ids, attention_mask, token_type_ids):
        out = self.s1(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return out.logits

    def forward_pair(self, input_ids_o, attention_mask_o, token_type_ids_o,
                     input_ids_f, attention_mask_f, token_type_ids_f, ling):
        """Both orderings through the shared backbone; S1 logits + S2 combo logits + gate."""
        logits_o, seq_o = self.encode(input_ids_o, attention_mask_o, token_type_ids_o)
        logits_f, seq_f = self.encode(input_ids_f, attention_mask_f, token_type_ids_f)
        p1 = F.softmax(logits_o, dim=-1)
        p2 = F.softmax(logits_f, dim=-1)
        # Signals: entropy (max of the two), reversal divergence KL(p1 || P.p2).
        e1 = -(p1 * torch.log(p1.clamp_min(1e-8))).sum(-1)
        e2 = -(p2 * torch.log(p2.clamp_min(1e-8))).sum(-1)
        ent = torch.maximum(e1, e2)
        p2_perm = p2[:, PERM]
        div = (p1 * (torch.log(p1.clamp_min(1e-8)) - torch.log(p2_perm.clamp_min(1e-8)))).sum(-1)
        # Pooled asymmetry between the two orderings' representations.
        h_diff = (seq_o[:, 0] - seq_f[:, 0]).norm(dim=-1) / (seq_o.shape[-1] ** 0.5)
        feats = torch.cat([ent.unsqueeze(-1), div.unsqueeze(-1), ling, h_diff.unsqueeze(-1)], dim=-1)
        g = torch.sigmoid(self.router(feats)).squeeze(-1)
        # System 2: cross-attend each ordering's CLS over the other ordering's tokens.
        q1, q2 = self.proj(seq_o[:, 0:1]), self.proj(seq_f[:, 0:1])
        kv1, kv2 = self.proj(seq_o), self.proj(seq_f)
        mask1, mask2 = attention_mask_o == 0, attention_mask_f == 0
        u1, _ = self.cross_attn(q1, kv2, kv2, key_padding_mask=mask2)
        u2, _ = self.cross_attn(q2, kv1, kv1, key_padding_mask=mask1)
        combo_logits = self.pair_fc(torch.cat([u1.squeeze(1), u2.squeeze(1)], dim=-1))
        return logits_o, logits_f, combo_logits, g, div

    def predict_pair(self, input_ids_o, attention_mask_o, token_type_ids_o,
                     input_ids_f, attention_mask_f, token_type_ids_f, ling, threshold,
                     route_rule="compat+gate"):
        """Hard routing at inference.

        route_rule:
          - "compat": deliberate iff S1's two argmax labels are reversal-
            incompatible (pred(A,B) != P.pred(B,A)) -- a deterministic,
            zero-cost router that guarantees SoftCons = 1.
          - "gate":   deliberate iff the learned gate g > threshold.
          - "compat+gate": both (compatibility fixes + budgeted accuracy help).
        """
        logits_o, logits_f, combo_logits, g, _ = self.forward_pair(
            input_ids_o, attention_mask_o, token_type_ids_o,
            input_ids_f, attention_mask_f, token_type_ids_f, ling,
        )
        combo = combo_logits.argmax(-1)
        lab_o, lab_f = torch.tensor(COMBOS, device=combo.device)[combo].t()
        pred_o_s1 = logits_o.argmax(-1)
        pred_f_s1 = logits_f.argmax(-1)
        perm = torch.tensor(PERM, device=g.device)
        # Official SoftCons definition: the first prediction must be a
        # reversible label (not NEGATIVE_OTHER) and the second must equal its
        # deterministic reverse.
        not_soft = (pred_o_s1 == 3) | (pred_f_s1 != perm[pred_o_s1])
        if route_rule == "compat":
            use_s2 = not_soft
        elif route_rule == "gate":
            use_s2 = g > threshold
        else:
            use_s2 = not_soft | (g > threshold)
        pred_o = torch.where(use_s2, lab_o, pred_o_s1)
        pred_f = torch.where(use_s2, lab_f, pred_f_s1)
        return pred_o, pred_f, use_s2.float(), not_soft


def load_split(path: str, tokenizer, max_length: int):
    """Returns tokenized rows, pair index (original, flipped), unpaired rows, ids, labels."""
    df = pd.read_csv(path)
    enc = tokenizer(
        list(df["text1"]), list(df["text2"]),
        padding="max_length", truncation=True, max_length=max_length, return_tensors="pt",
    )
    rows = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "token_type_ids": enc["token_type_ids"],
        "ids": list(df["instance_id"]),
        "y": torch.tensor([L2I[l] for l in df["label"]], dtype=torch.long),
    }
    pairs, unpaired = [], []
    for _, grp in df.groupby("pair_id", sort=False):
        idxs = grp.index.to_numpy()
        if len(grp) == 2:
            o = next(i for i in idxs if "original" in rows["ids"][i])
            f = next(i for i in idxs if "flipped" in rows["ids"][i])
            pairs.append((int(o), int(f)))
        else:
            unpaired.append(int(idxs[0]))
    # Linguistic features per pair (from the original instance's texts).
    pair_ling = torch.tensor(
        [linguistic_features(str(df["text1"][o]), str(df["text2"][o])) for o, _ in pairs],
        dtype=torch.float,
    )
    pair_target = torch.tensor([LABEL_TO_COMBO[rows["y"][o].item()] for o, _ in pairs], dtype=torch.long)
    return rows, pairs, pair_ling, pair_target, torch.tensor(unpaired, dtype=torch.long)


def batch_rows(rows, idxs, device):
    idxs = torch.as_tensor(idxs, dtype=torch.long)
    return (
        rows["input_ids"][idxs].to(device),
        rows["attention_mask"][idxs].to(device),
        rows["token_type_ids"][idxs].to(device),
    )


def official_score(prediction_path: Path, reference_path: str, output_dir: Path):
    from evaluation_functions import ScoringError, score_files
    try:
        report = score_files(
            reference_path=reference_path,
            prediction_path=str(prediction_path),
            output_dir=output_dir,
        )
    except ScoringError as exc:
        print(f"Official scoring skipped: {exc}")
        return None
    payload = report.to_dict()
    return {k: payload[k] for k in ("weighted_f1", "soft_cons", "hard_cons")}


@torch.no_grad()
def evaluate(model, dev, device, threshold, results_dir, partition, route_rule="compat+gate", score=True):
    model.eval()
    rows, pairs, pair_ling, _, unpaired = dev
    pred_by_id, s1_only_by_id, deliberate_by_id = {}, {}, {}
    diag = {"pairs": 0, "delib": 0, "incompat": 0, "s1_pair_ok": 0, "final_pair_ok": 0,
            "delib_s1_ok": 0, "delib_final_ok": 0}
    for k in range(0, len(pairs), 32):
        chunk = pairs[k:k + 32]
        o = torch.tensor([c[0] for c in chunk])
        f = torch.tensor([c[1] for c in chunk])
        io, ao, to = batch_rows(rows, o, device)
        inf, af, tf = batch_rows(rows, f, device)
        ling = pair_ling[k:k + 32].to(device)
        pred_o, pred_f, use_s2, incompat = model.predict_pair(
            io, ao, to, inf, af, tf, ling, threshold, route_rule)
        lo = model.forward_s1(io, ao, to).argmax(-1).cpu()
        lf = model.forward_s1(inf, af, tf).argmax(-1).cpu()
        y_o = rows["y"][o]
        y_f = rows["y"][f]
        for j in range(len(chunk)):
            pred_by_id[rows["ids"][o[j]]] = pred_o[j].item()
            pred_by_id[rows["ids"][f[j]]] = pred_f[j].item()
            s1_only_by_id[rows["ids"][o[j]]] = lo[j].item()
            s1_only_by_id[rows["ids"][f[j]]] = lf[j].item()
            deliberate_by_id[rows["ids"][o[j]]] = bool(use_s2[j].item())
            deliberate_by_id[rows["ids"][f[j]]] = bool(use_s2[j].item())
            diag["pairs"] += 1
            s1_ok = bool(lo[j] == y_o[j] and lf[j] == y_f[j])
            fin_ok = bool(pred_o[j].item() == y_o[j].item() and pred_f[j].item() == y_f[j].item())
            diag["s1_pair_ok"] += s1_ok
            diag["final_pair_ok"] += fin_ok
            if use_s2[j].item():
                diag["delib"] += 1
                diag["delib_s1_ok"] += s1_ok
                diag["delib_final_ok"] += fin_ok
            if incompat[j].item():
                diag["incompat"] += 1
    for k in range(0, len(unpaired), 64):
        u = unpaired[k:k + 64]
        iu, au, tu = batch_rows(rows, u, device)
        lu = model.forward_s1(iu, au, tu).argmax(-1)
        for j in range(len(u)):
            pred_by_id[rows["ids"][u[j]]] = lu[j].item()
            s1_only_by_id[rows["ids"][u[j]]] = lu[j].item()
            deliberate_by_id[rows["ids"][u[j]]] = False

    y_true = [rows["y"][i].item() for i in range(len(rows["y"]))]
    y_pred = [pred_by_id[i] for i in rows["ids"]]
    y_s1 = [s1_only_by_id[i] for i in rows["ids"]]
    n_delib = sum(1 for i in rows["ids"] if deliberate_by_id[i])

    report = classification_report(y_true, y_pred, target_names=LABELS, digits=3, output_dict=True, zero_division=0)
    report_s1 = classification_report(y_true, y_s1, target_names=LABELS, digits=3, output_dict=True, zero_division=0)
    print(f"\n--- dev (joint routing, gate>{threshold}) ---")
    print(classification_report(y_true, y_pred, target_names=LABELS, digits=3, zero_division=0))
    print(confusion_matrix(y_true, y_pred))
    print(f"deliberated instances: {n_delib}/{len(rows['ids'])} ({n_delib/len(rows['ids']):.1%})")
    print(f"pairs: {diag['pairs']} | S1 soft-inconsistent (official def): {diag['incompat']} | deliberated pairs: {diag['delib']}")
    print(f"pair-level correctness: S1 {diag['s1_pair_ok']}/{diag['pairs']} -> final {diag['final_pair_ok']}/{diag['pairs']}"
          f"  (among deliberated: S1 {diag['delib_s1_ok']} -> S2 {diag['delib_final_ok']}/{diag['delib']})")
    print(f"weighted F1 (sklearn): joint={report['weighted avg']['f1-score']:.4f}  S1-only={report_s1['weighted avg']['f1-score']:.4f}")

    metrics = {"weighted_f1_sklearn_joint": report["weighted avg"]["f1-score"],
               "weighted_f1_sklearn_s1_only": report_s1["weighted avg"]["f1-score"],
               "deliberated_instances": n_delib, "total_instances": len(rows["ids"]),
               "diagnostics": diag}
    if score:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        pred_path = Path(results_dir) / f"predictions_dev_{ts}.csv"
        with pred_path.open("w", encoding="utf-8") as fh:
            fh.write("instance_id,label\n")
            for iid in rows["ids"]:
                fh.write(f"{iid},{LABELS[pred_by_id[iid]]}\n")
        official = official_score(pred_path, args.reference_path, results_dir / f"{pred_path.stem}_official_scores")
        if official:
            print(f"Official: WF1={official['weighted_f1']:.4f} SoftCons={official['soft_cons']:.4f} HardCons={official['hard_cons']:.4f}")
            metrics.update({f"official_{k}": v for k, v in official.items()})
    model.train()
    return metrics


@torch.no_grad()
def dev_gates(model, dev, device) -> np.ndarray:
    """Gate values for all dev pairs (for the adaptive threshold)."""
    model.eval()
    rows, pairs, pair_ling = dev[0], dev[1], dev[2]
    gs = []
    for k in range(0, len(pairs), 32):
        chunk = pairs[k:k + 32]
        o = torch.tensor([c[0] for c in chunk])
        f = torch.tensor([c[1] for c in chunk])
        io, ao, to = batch_rows(rows, o, device)
        inf, af, tf = batch_rows(rows, f, device)
        ling = pair_ling[k:k + 32].to(device)
        _, _, _, g, _ = model.forward_pair(io, ao, to, inf, af, tf, ling)
        gs += g.tolist()
    return np.array(gs)


def main() -> None:
    global args
    args = parse_args()

    task_repo = Path(args.task_repo)
    if args.model_path is None:
        args.model_path = str(OUTPUT_ROOT / "models/deberta_v3_base_track1")
    if args.train_path is None:
        args.train_path = str(task_repo / "final_data/train/dico_nli_train_track1_participant_labeled.csv")
    if args.dev_path is None:
        args.dev_path = str(task_repo / "final_data/dev/dico_nli_dev_track1_participant_labeled.csv")
    if args.reference_path is None:
        args.reference_path = str(task_repo / "final_data/dev/dico_nli_dev_track1_reference.csv")
    if args.save_path is None:
        args.save_path = str(OUTPUT_ROOT / "models/baleco_track1")
    if args.results_dir is None:
        args.results_dir = str(OUTPUT_ROOT / "results/baleco_track1")
    if args.baseline_scores is None:
        candidates = sorted((OUTPUT_ROOT / "results/deberta_v3_base_track1").glob(
            "predictions_*_official_scores/scores.json"))
        args.baseline_scores = str(candidates[-1]) if candidates else None

    # Official scorer package lives at the task repository root.
    sys.path.insert(0, str(task_repo))

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    g = set_seed(args.seed)
    device = pick_device()
    print(f"Device: {device}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path)
    print("Loading train/dev ...")
    train = load_split(args.train_path, tokenizer, args.max_length)
    dev = load_split(args.dev_path, tokenizer, args.max_length)
    rows, pairs, pair_ling, pair_target, unpaired = train
    if args.smoke:
        pairs = pairs[:64]
        pair_ling, pair_target = pair_ling[:64], pair_target[:64]
        unpaired = unpaired[:16]
        args.num_epochs = 1
    print(f"train: {len(rows['y'])} rows, {len(pairs)} pairs, {len(unpaired)} singles | dev: {len(dev[0]['y'])} rows, {len(dev[1])} pairs")

    model = Baleco(args.model_path).to(device)

    # Default: System 1 stays exactly the trained fast path; only System 2 and
    # the router learn. (--tune_s1 unfreezes the backbone for the joint variant.)
    if not args.tune_s1:
        for p in model.s1.parameters():
            p.requires_grad_(False)
        print("System 1 frozen (training S2 + router only)")

    # Supervised gate targets: pairs where frozen System 1 is wrong on either
    # direction, or its two predictions are reversal-incompatible. With S1
    # frozen these are fixed labels: the gate learns a calibrated
    # "S1 fallibility detector" from S1's own training-set mistakes.
    z_train = None
    if not args.gate_routing:
        model.eval()
        z_train = torch.zeros(len(pairs))
        with torch.no_grad():
            for k in range(0, len(pairs), 64):
                chunk = pairs[k:k + 64]
                o = torch.tensor([c[0] for c in chunk])
                f = torch.tensor([c[1] for c in chunk])
                io, ao, to = batch_rows(rows, o, device)
                inf, af, tf = batch_rows(rows, f, device)
                lo = model.forward_s1(io, ao, to).argmax(-1).cpu()
                lf = model.forward_s1(inf, af, tf).argmax(-1).cpu()
                y_o = rows["y"][o]
                y_f = rows["y"][f]
                wrong = (lo != y_o) | (lf != y_f)
                perm = torch.tensor(PERM)
                not_soft = (lo == 3) | (lf != perm[lo])
                z_train[k:k + len(chunk)] = (wrong | not_soft).float()
        print(f"supervised gate: needs-deliberation pairs in train: {z_train.mean():.1%}")

    head_modules = [model.proj, model.cross_attn, model.pair_fc, model.router]
    groups = [{"params": [p for m in head_modules for p in m.parameters()],
               "lr": args.lr_heads, "weight_decay": args.wd}]
    if args.tune_s1:
        groups.insert(0, {"params": [p for p in model.s1.parameters() if p.requires_grad],
                           "lr": args.lr_backbone, "weight_decay": args.wd})
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-6)
    steps_per_epoch = (len(pairs) + args.batch_pairs - 1) // args.batch_pairs
    if args.tune_s1:
        steps_per_epoch += (len(unpaired) + 63) // 64
    total_steps = steps_per_epoch * args.num_epochs
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_pcrt), total_steps)

    log = []
    t0 = time.time()
    for epoch in range(args.num_epochs):
        perm = torch.randperm(len(pairs), generator=g).tolist()
        ep = {"l_route": 0.0, "l_s2": 0.0, "l_sing": 0.0, "l_rev": 0.0, "l_gate": 0.0, "gate": [], "steps": 0}
        model.train()
        for k in range(0, len(pairs), args.batch_pairs):
            idxs = perm[k:k + args.batch_pairs]
            chunk = [pairs[i] for i in idxs]
            if not chunk:
                break
            o = torch.tensor([c[0] for c in chunk])
            f = torch.tensor([c[1] for c in chunk])
            io, ao, to = batch_rows(rows, o, device)
            inf, af, tf = batch_rows(rows, f, device)
            ling = pair_ling[idxs].to(device)
            target = pair_target[idxs].to(device)
            y_o = rows["y"][o].to(device)
            y_f = rows["y"][f].to(device)

            logits_o, logits_f, combo_logits, gate, div = model.forward_pair(
                io, ao, to, inf, af, tf, ling)
            ce_s1 = 0.5 * (F.cross_entropy(logits_o, y_o, reduction="none")
                           + F.cross_entropy(logits_f, y_f, reduction="none"))
            ce_s2 = F.cross_entropy(combo_logits, target, reduction="none")
            # Gate: supervised on S1's needs-fix pairs (default), or by CE
            # consequences (the gate pays whichever head it routes to).
            if z_train is not None:
                l_route = F.binary_cross_entropy(gate, z_train[idxs].to(device))
            else:
                l_route = (gate * ce_s2 + (1 - gate) * ce_s1).mean()
            # Unweighted S2 CE: S2 learns on all pairs even while the gate is closed.
            l_s2 = ce_s2.mean()
            l_rev = div.mean()
            rho = gate.mean()
            l_gate = args.lambda_gate * F.relu(rho - args.gate_budget) ** 2
            loss = l_route + args.lambda2 * l_s2 + l_gate
            if args.tune_s1:
                loss = loss + args.lambda_rev * l_rev

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            ep["l_route"] += l_route.item(); ep["l_s2"] += l_s2.item(); ep["l_rev"] += l_rev.item()
            ep["l_gate"] += l_gate.item(); ep["gate"].append(rho.item()); ep["steps"] += 1

        for k in range(0, len(unpaired) if args.tune_s1 else 0, 64):
            u = unpaired[k:k + 64]
            iu, au, tu = batch_rows(rows, u, device)
            y_u = rows["y"][u].to(device)
            l_sing = F.cross_entropy(model.forward_s1(iu, au, tu), y_u)
            optimizer.zero_grad()
            l_sing.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            ep["l_sing"] += l_sing.item(); ep["steps"] += 1

        s = max(ep["steps"], 1)
        print(f"\n>>> Epoch {epoch + 1}/{args.num_epochs}  ({time.time() - t0:.0f}s elapsed)")
        print(f"    l_route={ep['l_route']/s:.4f} l_s2={ep['l_s2']/s:.4f} l_sing={ep['l_sing']/s:.4f} "
              f"l_rev={ep['l_rev']/s:.4f} l_gate={ep['l_gate']/s:.4f} gate={np.mean(ep['gate']):.3f}")
        if args.smoke:
            continue
        gates = dev_gates(model, dev, device)
        tau_q = float(np.quantile(gates, 1 - args.gate_budget))
        for tag, tau in [("tau_half", args.gate_threshold), ("tau_budget", tau_q)]:
            print(f"--- dev routing: rule={args.route_rule}, gate threshold {tau:.3f} ({tag}) ---")
            metrics = evaluate(model, dev, device, tau, results_dir, "dev", route_rule=args.route_rule)
            log.append({"epoch": epoch + 1, "threshold_tag": tag, "threshold": tau, "route_rule": args.route_rule,
                        "losses": {k: v / s for k, v in ep.items() if k != "gate"},
                        "gate_mean": float(np.mean(ep["gate"])), "metrics": metrics})

    if not args.smoke:
        save_dir = Path(args.save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        model.s1.save_pretrained(save_dir / "s1")
        tokenizer.save_pretrained(save_dir / "s1")
        torch.save(
            {k: v for k, v in model.state_dict().items() if k.startswith(("proj", "cross_attn", "pair_fc", "router"))},
            save_dir / "heads.pt",
        )
        (save_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
        print(f"\nSaved model to {save_dir}")

        (results_dir / "training_log.json").write_text(json.dumps(log, indent=2))
        print("\n==================== FINAL COMPARISON (Track 1 dev, official scorer) ====================")
        if args.baseline_scores and Path(args.baseline_scores).exists():
            base = json.loads(Path(args.baseline_scores).read_text())
            print(f"{'System':<32}{'Weighted F1':>13}{'SoftCons':>11}{'HardCons':>11}")
            print(f"{'DeBERTa-v3-base (System 1)':<32}{base['weighted_f1']:>13.4f}{base['soft_cons']:>11.4f}{base['hard_cons']:>11.4f}")
            scored = [e for e in log if "official_weighted_f1" in e["metrics"]]
            # Primary operating point: last epoch at the budget threshold
            # (threshold set by the deliberation budget, not tuned on dev).
            last_budget = [e for e in scored if e["threshold_tag"] == "tau_budget"][-1]
            m = last_budget["metrics"]
            print(f"{'BALECO (budget op. point)':<32}{m['official_weighted_f1']:>13.4f}{m['official_soft_cons']:>11.4f}{m['official_hard_cons']:>11.4f}")
            print(f"  (epoch {last_budget['epoch']}, threshold {last_budget['threshold']:.3f} [budget], "
                  f"deliberated {m['deliberated_instances']}/{m['total_instances']}, "
                  f"S1-only sklearn WF1 {m['weighted_f1_sklearn_s1_only']:.4f})")
            # Dev-selected point: best SoftCons with WF1 within 0.003 of baseline.
            near = [e for e in scored if e["metrics"]["official_weighted_f1"] >= base["weighted_f1"] - 0.003]
            if near:
                best = max(near, key=lambda e: (e["metrics"]["official_soft_cons"], e["metrics"]["official_hard_cons"]))
                mb = best["metrics"]
                print(f"{'BALECO (dev-selected)':<32}{mb['official_weighted_f1']:>13.4f}{mb['official_soft_cons']:>11.4f}{mb['official_hard_cons']:>11.4f}")
                print(f"  (epoch {best['epoch']}, threshold {best['threshold']:.3f} [{best['threshold_tag']}], "
                      f"deliberated {mb['deliberated_instances']}/{mb['total_instances']})")
        else:
            print("(baseline comparison unavailable: run train_deberta.py first)")
        print("==========================================================================================")


if __name__ == "__main__":
    main()
