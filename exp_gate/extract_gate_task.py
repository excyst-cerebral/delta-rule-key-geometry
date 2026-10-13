"""Part B of exp_gate/PROTOCOL.md: gate-aware key-only quantities for the queried fact of every pre-registered recall prompt.
usage: GDN_MD=<model dir> NT=<threads> python -I exp_gate/extract_gate_task.py <shard> <nshards> <prompts_per_load>
Prompts are the same as exp_task/extract_task.py (determined by (n, j)); outputs exp_gate/out/<model>_task_shard<k>.npz with, per prompt and for the three
span tokens (key, "is", value) of the queried fact, per layer and head:
  Dg   = || S_T k_s - A S_s k_s || / || S_s k_s ||   (key-only recurrence, REAL gates, values = keys; T = last prefix token)
  Rg   = || S_T k_s || / || S_s k_s ||               (retention)
  logA = log of the product of the real decay gates between s and T."""
import os, sys, time, random
HERE = "."; sys.path.insert(0, HERE); sys.path.insert(0, HERE + "/exp_task")
import numpy as np, torch
import gdn_extract as G
G.PROBES, G.DELTAS = [], []; torch.set_num_threads(int(os.environ.get("NT", "2")))
from recall_lib import build_vocab, make_prompt
from transformers import AutoTokenizer
shard, nsh, per = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); LOADS = [4, 8, 16, 24]
name = os.path.basename(G.MD.rstrip("/")); os.makedirs(HERE + "/exp_gate/out", exist_ok=True)
tok = AutoTokenizer.from_pretrained(G.MD); vocab = build_vocab(tok); bos = tok.bos_token_id
H, NL, DK = G.H, G.NL, G.DK
keys, recs = [], []; t0 = time.time()
for n in LOADS:
    for j in range(per):
        pid = LOADS.index(n) * per + j
        if pid % nsh != shard: continue
        rng = random.Random(10_000 * n + j); p = make_prompt(rng, vocab, n, bos)
        qpos = p["qpos"]; s0, s1 = p["spans"][p["q"]]; span = list(range(s0, s1)); T = qpos
        Dg = np.zeros((3, NL, H), np.float32); Rg = np.zeros((3, NL, H), np.float32); lA = np.zeros((3, NL, H), np.float32)
        with torch.no_grad():
            _, col = G.forward(torch.tensor(p["ids"]), collect_layers=set(range(NL)))
            for li in range(NL):
                d = col[li]; K = torch.from_numpy(d["K"][:T]).float(); al = torch.from_numpy(d["alpha"][:T]).float(); be = torch.from_numpy(d["beta"][:T]).float()
                S = torch.zeros(H, DK, DK); r0 = {}
                for t in range(T):
                    S = S * al[t][:, None, None]
                    e = torch.bmm(S, K[t][:, :, None])[:, :, 0] - K[t]
                    S = S - be[t][:, None, None] * e[:, :, None] * K[t][:, None, :]
                    if t in span: r0[t] = torch.bmm(S, K[t][:, :, None])[:, :, 0].clone()
                cs = torch.cat([torch.zeros(1, H), torch.cumsum(torch.log(al), 0)])
                for i, s in enumerate(span):
                    r1 = torch.bmm(S, K[s][:, :, None])[:, :, 0]; lA_s = cs[T] - cs[s + 1]; A = torch.exp(lA_s)
                    Dg[i, li] = ((r1 - A[:, None] * r0[s]).norm(dim=1) / r0[s].norm(dim=1)).numpy(); Rg[i, li] = (r1.norm(dim=1) / r0[s].norm(dim=1)).numpy(); lA[i, li] = lA_s.numpy()
        keys.append((n, j)); recs.append((Dg, Rg, lA))
        if len(recs) % 5 == 0: print(f"[{name} shard {shard}] {len(recs)} prompts {time.time() - t0:.0f}s", flush=True)
np.savez(f"{HERE}/exp_gate/out/{name}_task_shard{shard}.npz", nj=np.array(keys), Dg=np.stack([r[0] for r in recs]), Rg=np.stack([r[1] for r in recs]), logA=np.stack([r[2] for r in recs]))
print("done", name, shard, len(recs), f"{time.time() - t0:.0f}s")
