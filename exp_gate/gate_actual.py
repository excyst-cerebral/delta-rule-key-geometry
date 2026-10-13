"""Part A of exp_gate/PROTOCOL.md: interference in the ACTUAL recurrence of a trained Gated DeltaNet against gate-aware / gate-free key-only predictors.
usage: GDN_MD=<model dir> NT=<threads> python -I exp_gate/gate_actual.py [n_sequences_per_domain]
Writes exp_gate/out/<model>_actual.npz.  Torch only for the recurrences; numpy for the statistics."""
import os, sys, time
HERE = "."; sys.path.insert(0, HERE)
import numpy as np, torch
import gdn_extract as G
torch.set_num_threads(int(os.environ.get("NT", "5")))
from transformers import AutoTokenizer

PROBES = [64, 128, 192, 256, 320, 384, 448, 512]; DELTAS = [16, 64, 256, 512]; WFREE = 32
G.PROBES, G.DELTAS = PROBES, DELTAS
NL, H, DK, T = G.NL, G.H, G.DK, G.T
name = os.path.basename(G.MD.rstrip("/")); os.makedirs(HERE + "/exp_gate/out", exist_ok=True)
nper = int(sys.argv[1]) if len(sys.argv) > 1 else 4
valid = lambda s, d: s + d <= T - 1


def key_recurrence(K, al, be):
    """key-only recurrence with the real gates and values = keys.  Returns snapshots S_t k_s for all (s, d) of the grid and the surprises e_t."""
    sched = {}
    for s in PROBES:
        sched.setdefault(s, []).append((s, 0))
        for d in DELTAS:
            if valid(s, d): sched.setdefault(s + d, []).append((s, d))
    S = torch.zeros(H, DK, DK); snaps, E = {}, []
    for t in range(K.shape[0]):
        S = S * al[t][:, None, None]
        e = torch.bmm(S, K[t][:, :, None])[:, :, 0] - K[t]
        S = S - be[t][:, None, None] * e[:, :, None] * K[t][:, None, :]
        E.append(e)
        for (s, d) in sched.get(t, []): snaps[(s, d)] = torch.bmm(S, K[s][:, :, None])[:, :, 0].clone()
    return snaps, torch.stack(E)


def free_recurrence(K, s):
    """alpha = beta = 1, values = keys, started empty WFREE tokens before s (the idealised memory of the paper); snapshots S_t k_s at t = s and s + d."""
    S = torch.zeros(H, DK, DK); want = {s + d: d for d in DELTAS if valid(s, d)}; snaps = {}
    for t in range(s - WFREE + 1, s + max(want, default=s) - s + 1):
        e = torch.bmm(S, K[t][:, :, None])[:, :, 0] - K[t]
        S = S - e[:, :, None] * K[t][:, None, :]
        if t == s: snaps[0] = torch.bmm(S, K[s][:, :, None])[:, :, 0].clone()
        if t in want: snaps[want[t]] = torch.bmm(S, K[s][:, :, None])[:, :, 0].clone()
    return snaps


def nrm(x): return np.linalg.norm(x, axis=-1)


tok = AutoTokenizer.from_pretrained(G.MD); data = G.load_sequences(tok, nper)
seqs = [(dom, s) for dom, ss in data.items() for s in ss]; nS, nP, nD = len(seqs), len(PROBES), len(DELTAS)
out = {k: np.full((nS, NL, H, nP, nD), np.nan, np.float32) for k in ("Dact", "Dgate", "Dshuf", "Dfree", "A", "Bsum", "Bkey", "Rgate", "Ract", "lemma")}
hl = np.zeros((nS, NL, H), np.float32); t0 = time.time()
with torch.no_grad():
    for si, (dom, seq) in enumerate(seqs):
        _, col = G.forward(torch.tensor(seq), collect_layers=set(range(NL)))
        for li in range(NL):
            d = col[li]; K = torch.from_numpy(d["K"]).float(); al = torch.from_numpy(d["alpha"]).float(); be = torch.from_numpy(d["beta"]).float()
            ksn, Ek = key_recurrence(K, al, be); Ek = Ek.numpy().astype(np.float64)
            perm = torch.from_numpy(np.random.default_rng(1000 * si + li).permutation(K.shape[0])); ssn, _ = key_recurrence(K[perm], al, be)   # control: real gates, keys permuted over positions
            hl[si, li] = np.median(np.log(2) / np.maximum(-np.log(d["alpha"]), 1e-9), axis=0)
            cs = np.concatenate([np.zeros((1, H)), np.cumsum(np.log(d["alpha"]), 0)]); Kn = d["K"]; be_n = d["beta"].astype(np.float64)
            for pi, s in enumerate(PROBES):
                fsn = free_recurrence(K, s); r0a = d["snaps"][(s, 0)]; r0g = ksn[(s, 0)].numpy().astype(np.float64); r0f = fsn[0].numpy().astype(np.float64)
                for di, dl in enumerate(DELTAS):
                    if not valid(s, dl): continue
                    A = np.exp(cs[s + dl + 1] - cs[s + 1])                                                                           # (H,) product of alpha over (s, s+dl]
                    r1a = d["snaps"][(s, dl)]; r1g = ksn[(s, dl)].numpy().astype(np.float64); r1f = fsn[dl].numpy().astype(np.float64)
                    out["Dact"][si, li, :, pi, di] = nrm(r1a - A[:, None] * r0a) / nrm(r0a)
                    out["Dgate"][si, li, :, pi, di] = nrm(r1g - A[:, None] * r0g) / nrm(r0g)
                    r0s = ssn[(s, 0)].numpy().astype(np.float64); r1s = ssn[(s, dl)].numpy().astype(np.float64); out["Dshuf"][si, li, :, pi, di] = nrm(r1s - A[:, None] * r0s) / nrm(r0s)
                    out["Dfree"][si, li, :, pi, di] = nrm(r1f - r0f) / nrm(r0f)
                    out["Ract"][si, li, :, pi, di] = nrm(r1a) / nrm(r0a); out["Rgate"][si, li, :, pi, di] = nrm(r1g) / nrm(r0g)
                    out["A"][si, li, :, pi, di] = A
                    ts = np.arange(s + 1, s + dl + 1); Gm = np.einsum("uhd,hd->uh", Kn[ts], Kn[s])                                  # (dl, H) <k_u, k_s>
                    out["Bsum"][si, li, :, pi, di] = be_n[ts].sum(0); out["Bkey"][si, li, :, pi, di] = (be_n[ts] * np.abs(Gm)).sum(0)
                    if dl == 64:                                                                                                       # interference identity on the key-only recurrence
                        al_n = d["alpha"]; w = np.array([np.prod(al_n[u + 1:s + dl + 1], axis=0) for u in ts])                         # prod_{v>u} alpha_v, (dl, H)
                        pred = A[:, None] * r0g - np.einsum("uh,uhd->hd", w * be_n[ts] * Gm, Ek[ts])
                        out["lemma"][si, li, :, pi, di] = nrm(pred - r1g) / nrm(r0g)
        print(f"[{name}] sequence {si + 1}/{nS} ({dom}) {time.time() - t0:.0f}s  max key-only identity residual so far {np.nanmax(out['lemma']):.2e}", flush=True)
np.savez(f"{HERE}/exp_gate/out/{name}_actual.npz", probes=PROBES, deltas=DELTAS, dom=np.array([d for d, _ in seqs]), halflife=hl, **out)
print("saved", name, nS, "sequences", f"{time.time() - t0:.0f}s")
