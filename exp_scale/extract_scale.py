"""Scale replication (exp_scale/PROTOCOL.md): one forward pass per recall prompt records the outcome, the gate-free psi (as exp_task/extract_task.py), the gate-aware
key-only quantities (as exp_gate/extract_gate_task.py) and the per-token decay half-lives (for the long-memory head definition).  Resumable.
usage: GDN_MD=model_gdn13b NT=10 python -I exp_scale/extract_scale.py <prompts_per_load>   -> exp_scale/out/<model>_scale.npz"""
import os, sys, time, random
HERE = "."; sys.path.insert(0, HERE); sys.path.insert(0, HERE + "/exp_task")
import numpy as np, torch
import gdn_extract as G
G.PROBES, G.DELTAS = [], []; torch.set_num_threads(int(os.environ.get("NT", "10")))
from recall_lib import build_vocab, make_prompt
from transformers import AutoTokenizer
per = int(sys.argv[1]); LOADS = [4, 8, 16, 24]; TMAX = 100
name = os.path.basename(G.MD.rstrip("/")); OUT = f"{HERE}/exp_scale/out/{name}_scale.npz"; os.makedirs(os.path.dirname(OUT), exist_ok=True)
tok = AutoTokenizer.from_pretrained(G.MD); vocab = build_vocab(tok); bos = tok.bos_token_id; H, NL, DK = G.H, G.NL, G.DK
FIELDS = ("meta", "psi_span", "psi_all", "psi_last", "mlog", "Dg", "Rg", "logA", "hl")
acc = {k: [] for k in FIELDS}; done = set()
if os.path.exists(OUT):                                                   # resume
    z = np.load(OUT); acc = {k: list(z[k]) for k in FIELDS}; done = {(int(r[0]), int(r[1])) for r in z["meta"]}; print("resumed", len(done), "prompts", flush=True)
t0 = time.time(); k_new = 0


def save():
    np.savez(OUT + ".tmp.npz", **{k: np.stack(v) for k, v in acc.items()}); os.replace(OUT + ".tmp.npz", OUT)


for n in LOADS:
    for j in range(per):
        if (n, j) in done: continue
        p = make_prompt(random.Random(10_000 * n + j), vocab, n, bos); qpos = p["qpos"]; s0, s1 = p["spans"][p["q"]]; span = list(range(s0, s1)); T = qpos
        with torch.no_grad():
            logits, col = G.forward(torch.tensor(p["ids"]), collect_layers=set(range(NL)))
            last = logits[-1]; cl = torch.tensor([float(last[t]) for t in p["cands"]]); lsm = torch.log_softmax(cl, 0); ti = p["cands"].index(p["target"])
            fc = int(cl.argmax()) == ti; rank = int((cl > cl[ti]).sum()) + 1
            psi_span = np.zeros((3, NL, H), np.float32); psi_all = np.zeros((NL, H), np.float32); psi_last = np.zeros((NL, H), np.float32); mlog = np.zeros((NL, H), np.float32)
            Dg = np.zeros((3, NL, H), np.float32); Rg = np.zeros((3, NL, H), np.float32); lA = np.zeros((3, NL, H), np.float32); hl = np.full((NL, H, TMAX), np.nan, np.float32)
            for li in range(NL):
                d = col[li]; K = torch.from_numpy(d["K"][:T]).float(); al = torch.from_numpy(d["alpha"][:T]).float(); be = torch.from_numpy(d["beta"][:T]).float()
                for h in range(H):                                          # gate-free psi (alpha = beta = 1), exactly as exp_task/extract_task.py
                    Kh = K[:, h]; Gm = Kh @ Kh.T; L = torch.tril(Gm, -1); A = torch.eye(T) + L
                    X = torch.linalg.solve_triangular(A, Kh, upper=False, unitriangular=True); TK = L.T @ X; psi = TK.norm(dim=1)
                    psi_span[:, li, h] = psi[s0:s1].numpy(); psi_all[li, h] = float(psi.mean()); psi_last[li, h] = float(psi[-8:].mean()); mlog[li, h] = float(torch.log(al[:, h]).mean())
                    hl[li, h, :T] = (np.log(2) / np.maximum(-torch.log(al[:, h]).numpy(), 1e-9))
                S = torch.zeros(H, DK, DK); r0 = {}                         # gate-aware key-only recurrence: real gates, values = keys, as exp_gate/extract_gate_task.py
                for t in range(T):
                    S = S * al[t][:, None, None]
                    e = torch.bmm(S, K[t][:, :, None])[:, :, 0] - K[t]
                    S = S - be[t][:, None, None] * e[:, :, None] * K[t][:, None, :]
                    if t in span: r0[t] = torch.bmm(S, K[t][:, :, None])[:, :, 0].clone()
                cs = torch.cat([torch.zeros(1, H), torch.cumsum(torch.log(al), 0)])
                for i, s in enumerate(span):
                    r1 = torch.bmm(S, K[s][:, :, None])[:, :, 0]; lA_s = cs[T] - cs[s + 1]; A_ = torch.exp(lA_s)
                    Dg[i, li] = ((r1 - A_[:, None] * r0[s]).norm(dim=1) / r0[s].norm(dim=1)).numpy(); Rg[i, li] = (r1.norm(dim=1) / r0[s].norm(dim=1)).numpy(); lA[i, li] = lA_s.numpy()
        acc["meta"].append(np.array([n, j, p["age"], qpos, p["q"], fc, rank, float(lsm[ti]), int(last.argmax()) == p["target"], float(torch.log_softmax(last, -1)[p["target"]])], np.float64))
        for k, v in (("psi_span", psi_span), ("psi_all", psi_all), ("psi_last", psi_last), ("mlog", mlog), ("Dg", Dg), ("Rg", Rg), ("logA", lA), ("hl", hl)): acc[k].append(v)
        k_new += 1
        if k_new % 20 == 0: save(); print(f"[{name}] {len(acc['meta'])} prompts  {time.time() - t0:.0f}s  acc so far {np.mean([m[5] for m in acc['meta']]):.3f}", flush=True)
save(); print("done", name, len(acc["meta"]), "prompts", f"{time.time() - t0:.0f}s")
