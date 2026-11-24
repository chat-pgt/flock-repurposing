"""E5: fine-tune on treats with uniform or disease-only negatives, then score the held-out fold as in E1.

Each (fold, arm) starts from the released checkpoint, trains on the fold's training treats facts for a fixed
number of epochs (3 for ULTRA, 1 for FLOCK), and is scored on the fold's held-out edges after every epoch,
exactly as E1 scores them. Shards use E1's format, with condition `ft_<arm>_ep<epoch>`. Walks are reseeded
before scoring, so FLOCK's scoring walks match E1's for the same seed and fold, and both arms see the same
training batches and walks.

Usage, from experiments/:
    python -m e5_negatives.finetune --model ultra --ultra-repo ULTRA --output-dir runs
    python -m e5_negatives.finetune --model flock --folds 0 --output-dir runs
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from common.hetionet import EDGES_PATH, SPLIT_PATH, TREATS, read_edges, read_split
from e5_negatives import trainable_rspmm
from e5_negatives.negatives import ARMS, NegativeSampler, batches, training_indications
from gpu.graphs import encode_triples, training_mask
from gpu.models import build_data, load_flock, load_ultra, peak_gpu_gb, run_metadata, seed_everything
from gpu.run_entity import run_fold, save_shard

# Published fine-tuning settings (src_entity/config/finetune_transductive, ULTRA config/transductive).
PUBLISHED = {
    "ultra": {"batch_size": 8, "micro_batch": 8, "lr": 5e-4, "n_negative": 256, "temperature": 1.0},
    # FLOCK trains at n = 512 instead of 1024 to keep a run near an hour, with micro-batches of one query
    # accumulated to the published batch of 2 and a checkpointed sequence network, to fit a 16 GB T4.
    "flock": {"batch_size": 2, "micro_batch": 1, "lr": 5e-4, "n_negative": 256, "temperature": 1.0},
}
FLOCK_TRAIN_WALKS = 512
# Fixed in advance and all reported. ULTRA is cheap enough to score a learning curve; FLOCK is a pilot.
EPOCHS = {"ultra": 3, "flock": 1}
LOG_EVERY = 25


def adversarial_bce(pred, temperature):
    """The released training loss: binary cross-entropy with self-adversarial negative weights."""
    target = torch.zeros_like(pred)
    target[:, 0] = 1
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    weight = torch.ones_like(pred)
    with torch.no_grad():
        weight[:, 1:] = F.softmax(pred[:, 1:] / temperature, dim=-1)
    return ((loss * weight).sum(dim=-1) / weight.sum(dim=-1)).mean()


def train(model, data, optimizer, steps, sampler, drugs, diseases, treats, settings, device, max_steps=None):
    """Run one epoch of optimizer steps and return the losses and the time taken."""
    model.train()
    losses, start = [], time.perf_counter()
    for step, rows in enumerate(steps):
        if max_steps is not None and step >= max_steps:
            break
        heads, tails = drugs[rows], diseases[rows]
        candidates = np.concatenate([tails[:, None], sampler.sample(heads)], axis=1)
        batch = np.stack(
            [np.repeat(heads[:, None], candidates.shape[1], axis=1), candidates, np.full_like(candidates, treats)],
            axis=-1,
        )
        batch = torch.from_numpy(batch).long().to(device)
        total = 0.0
        for micro in torch.split(batch, settings["micro_batch"]):
            loss = adversarial_bce(model(data, micro), settings["temperature"]) * len(micro) / len(batch)
            loss.backward()
            total += loss.item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(total)
        if step % LOG_EVERY == 0:
            print(
                f"  step {step}: loss {total:.4f}, {(time.perf_counter() - start) / (step + 1):.2f} s/step, "
                f"peak GPU {peak_gpu_gb(device)}",
                flush=True,
            )
    model.eval()
    return np.array(losses), time.perf_counter() - start


class Checkpointed(torch.nn.Module):
    """Recompute a block in the backward pass instead of keeping its activations. Outputs are unchanged."""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        if not torch.is_grad_enabled():
            return self.block(x)
        return checkpoint(self.block, x, use_reentrant=False)


def load_model(name, device, ultra_repo, walk_num=None, ensemble=None):
    if name == "flock":
        model, settings = load_flock("entity", device, walk_num, ensemble)
        # Without this, one training query at n = 512 keeps about 12 GB of activations, mostly here.
        for index, block in enumerate(model.net):
            model.net[index] = Checkpointed(block)
        return model, settings
    model, settings = load_ultra(ultra_repo, device)
    trainable_rspmm.install()
    return model, settings


def shard_path(args, arm, epoch, fold):
    return args.output_dir / f"entity_{args.model}_ft_{arm}_ep{epoch}_tail_seed{args.seed}_fold{fold}.npz"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["flock", "ultra"], required=True)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, help="Training epochs, each scored (default: 3 for ULTRA, 1 for FLOCK)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deadline", type=float, help="Unix time after which no new fold starts")
    parser.add_argument("--ultra-repo", type=Path)
    parser.add_argument("--edges", type=Path, default=EDGES_PATH)
    parser.add_argument("--split", type=Path, default=SPLIT_PATH)
    parser.add_argument("--max-steps", type=int, help="Stop training early, for smoke tests only")
    parser.add_argument("--max-queries", type=int, help="Score only this many queries per fold, for smoke tests")
    parser.add_argument("--train-walk-num", type=int, default=FLOCK_TRAIN_WALKS, help="FLOCK's n during training")
    parser.add_argument("--n-negative", type=int, help="Override the number of negatives, for smoke tests only")
    parser.add_argument("--walk-num", type=int, help="Override FLOCK's scoring n, for smoke tests only")
    parser.add_argument("--ensemble", type=int, help="Override FLOCK's scoring P, for smoke tests only")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arms = args.arms.split(",")
    if not set(arms) <= set(ARMS):
        raise SystemExit(f"arms must be among {ARMS}")
    settings = dict(PUBLISHED[args.model])
    epochs = args.epochs or EPOCHS[args.model]
    if args.n_negative is not None:
        settings["n_negative"] = args.n_negative

    graph = encode_triples(read_edges(args.edges))
    split = read_split(args.split)
    index = {name: position for position, name in enumerate(graph.entities)}
    drugs = np.array([index[name] for name in split.drugs.tolist()])
    diseases = np.array([index[name] for name in split.diseases.tolist()])
    disease_ids = np.array([position for position, name in enumerate(graph.entities) if name.startswith("Disease::")])
    known = defaultdict(set)
    for drug, disease in zip(drugs.tolist(), diseases.tolist(), strict=True):
        known[drug].add(disease)
    treats = graph.relation_names.index(TREATS)
    split_ids = (drugs, diseases, split.folds, known, disease_ids, treats)
    score_args = SimpleNamespace(direction="tail", max_queries=args.max_queries)
    device = torch.device(args.device)

    for fold in [int(fold) for fold in args.folds.split(",")]:
        held = split.folds == fold
        train_rows = np.flatnonzero(~held)
        indications = training_indications(drugs, diseases, train_rows)
        data = None
        for arm in arms:
            if shard_path(args, arm, epochs, fold).exists():
                print(f"skip {shard_path(args, arm, epochs, fold).name}", flush=True)
                continue
            if args.deadline is not None and time.time() > args.deadline:
                print(f"time budget reached before {shard_path(args, arm, epochs, fold).name}", flush=True)
                return
            # Load first: ULTRA's graph builder is importable only once its repository is on the path.
            model, model_settings = load_model(args.model, device, args.ultra_repo, args.walk_num, args.ensemble)
            if data is None:
                keep = training_mask(graph, drugs[held], diseases[held], treats)
                data = build_data(graph, keep, args.model, device).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"])
            sampler = NegativeSampler(
                arm,
                len(graph.entities),
                disease_ids,
                indications,
                settings["n_negative"],
                np.random.default_rng([args.seed, fold, ARMS.index(arm)]),
            )
            order = np.random.default_rng([args.seed, fold])  # the same batches for both arms
            if str(device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            print(
                f"{arm} arm, fold {fold}: training on {len(train_rows)} treats facts for {epochs} epochs",
                flush=True,
            )
            steps_so_far = 0
            for epoch in range(1, epochs + 1):
                if args.model == "flock":
                    model.N = args.train_walk_num
                seed_everything(1000 * args.seed + fold + 100_000 * (epoch - 1))
                losses, train_seconds = train(
                    model,
                    data,
                    optimizer,
                    batches(train_rows, settings["batch_size"], 1, order),
                    sampler,
                    drugs,
                    diseases,
                    treats,
                    settings,
                    device,
                    args.max_steps,
                )
                steps_so_far += len(losses)
                if args.model == "flock":
                    model.N = model_settings["walk_num"]
                train_peak = peak_gpu_gb(device)
                seed_everything(1000 * args.seed + fold)  # E1's scoring walks
                start = time.perf_counter()
                with torch.no_grad():
                    arrays = run_fold(model, data, graph, split_ids, fold, score_args, device)
                seconds = time.perf_counter() - start
                info = {
                    **run_metadata(
                        device, model=args.model, direction="tail", condition=f"ft_{arm}_ep{epoch}", **model_settings
                    ),
                    "seed": args.seed,
                    "fold": fold,
                    "arm": arm,
                    "epoch": epoch,
                    "epochs": epochs,
                    "train_settings": settings,
                    "train_walk_num": args.train_walk_num if args.model == "flock" else None,
                    "train_steps": len(losses),
                    "train_steps_total": steps_so_far,
                    "train_seconds": train_seconds,
                    "train_peak_gpu_gb": train_peak,
                    "seconds": seconds,
                    "peak_gpu_gb": peak_gpu_gb(device),
                }
                save_shard(
                    shard_path(args, arm, epoch, fold), metadata=np.array(json.dumps(info)), train_loss=losses, **arrays
                )
                print(
                    f"{shard_path(args, arm, epoch, fold).name}: {len(losses)} steps, {train_seconds / 60:.1f} min, "
                    f"loss "
                    f"{losses[:10].mean():.4f} -> {losses[-10:].mean():.4f} | {len(arrays['edge_rows'])} edges, "
                    f"MRR {arrays['reciprocal_rank'].mean():.3f}, Hits@10 {arrays['hits10'].mean():.3f}, "
                    f"{seconds / len(arrays['query_entities']):.2f} s/query, peak GPU {train_peak}",
                    flush=True,
                )
            del model, optimizer
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
        del data


if __name__ == "__main__":
    main()
