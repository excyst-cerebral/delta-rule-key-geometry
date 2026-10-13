"""Run recall prompts through a trained GDN and record, per prompt, the outcome and the key-only certificate quantities.  Torch only (no numpy linear algebra).
usage: GDN_MD=<dir> python extract_task.py <shard> <nshards> <prompts_per_load>"""
import os, sys, time, json, random
HERE = "."; sys.path.insert(0, HERE); sys.path.insert(0, HERE + "/exp_task")
import numpy as np, torch
import gdn_extract as G
G.PROBES, G.DELTAS = [], []; torch.set_num_threads(1)
from recall_lib import build_vocab, make_prompt
from transformers import AutoTokenizer
shard, nsh, per = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); LOADS = [4, 8, 16, 24]
name = os.path.basename(G.MD.rstrip("/")); os.makedirs(HERE + "/exp_task/out", exist_ok=True)
tok = AutoTokenizer.from_pretrained(G.MD); vocab = build_vocab(tok); bos = tok.bos_token_id
H, NL = G.H, G.NL
recs = []; t0 = time.time()
for n in LOADS:
    for j in range(per):
        pid = LOADS.index(n) * per + j
        if pid % nsh != shard: continue
        rng = random.Random(10_000 * n + j); p = make_prompt(rng, vocab, n, bos)       # prompt determined by (n, j), independent of sharding
        with torch.no_grad():
            logits, col = G.forward(torch.tensor(p["ids"]), collect_layers=set(range(NL)))
            last = logits[-1]; cl = torch.tensor([float(last[t]) for t in p["cands"]]); lsm = torch.log_softmax(cl, 0); ti = p["cands"].index(p["target"])
            fc = int(cl.argmax()) == ti; rank = int((cl > cl[ti]).sum()) + 1
            qpos = p["qpos"]; s0, s1 = p["spans"][p["q"]]; T = qpos                      # prefix = everything before the query tokens
            psi_span = np.zeros((3, NL, H), np.float32); psi_all = np.zeros((NL, H), np.float32); mlog = np.zeros((NL, H), np.float32); psi_last = np.zeros((NL, H), np.float32)
            for li in range(NL):
                d = col[li]; K = torch.from_numpy(d["K"][:T]).float(); al = torch.from_numpy(d["alpha"][:T]).float()
                for h in range(H):
                    Kh = K[:, h]; Gm = Kh @ Kh.T; L = torch.tril(Gm, -1); A = torch.eye(T) + L
                    X = torch.linalg.solve_triangular(A, Kh, upper=False, unitriangular=True); TK = L.T @ X; psi = TK.norm(dim=1)
                    psi_span[:, li, h] = psi[s0:s1].numpy(); psi_all[li, h] = float(psi.mean()); psi_last[li, h] = float(psi[-8:].mean()); mlog[li, h] = float(torch.log(al[:, h]).mean())
        recs.append(dict(n=n, j=j, age=p["age"], qpos=qpos, q=p["q"], fc=fc, rank=rank, cand_logp=float(lsm[ti]), top1_vocab=int(last.argmax()) == p["target"], logp_vocab=float(torch.log_softmax(last, -1)[p["target"]]),
                         psi_span=psi_span, psi_all=psi_all, psi_last=psi_last, mlog=mlog))
        if len(recs) % 5 == 0: print(f"[{name} shard {shard}] {len(recs)} prompts {time.time()-t0:.0f}s", flush=True)
meta = np.array([[r["n"], r["j"], r["age"], r["qpos"], r["q"], r["fc"], r["rank"], r["cand_logp"], r["top1_vocab"], r["logp_vocab"]] for r in recs], np.float64)
np.savez(f"{HERE}/exp_task/out/{name}_shard{shard}.npz", meta=meta, psi_span=np.stack([r["psi_span"] for r in recs]), psi_all=np.stack([r["psi_all"] for r in recs]), psi_last=np.stack([r["psi_last"] for r in recs]), mlog=np.stack([r["mlog"] for r in recs]))
print("done", name, shard, len(recs), f"{time.time()-t0:.0f}s", flush=True)
