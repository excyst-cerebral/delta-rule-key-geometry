"""Common-cause test: pair head h's keys with ANOTHER head's values (same layer, same positions).
gap_x = (global-shuffle error - error with the real positional pairing) for each value source; ratio = gap_other / gap_own.
Also: other head's own error scale, and the same with values taken from a different LAYER's head at the same positions."""
import sys, json
sys.path.insert(0, ".")
import numpy as np
import torch
import gdn_extract as g
from gdn_kv import keys_values
from gdn_kv_tok import err_op, rms

NS = [32, 64]
ENDS = list(range(255, 1024, 64))
rng = np.random.default_rng(8)

if __name__ == "__main__":
    tok = g.AutoTokenizer.from_pretrained(g.MD)
    data = g.load_sequences(tok, n_per_domain=3)
    per = json.load(open("results_gdn.json"))["per_head"]
    long_heads = {d: {tuple(map(int, k.split("|")[1:])) for k, v in per.items() if k.startswith(d + "|") and v["p256_A"] > 0.5} for d in data}
    unitv = lambda V: V / np.maximum(np.linalg.norm(V, axis=-1, keepdims=True), 1e-12)
    res = {n: {k: [] for k in ("own_real", "own_shuf", "oh_real", "oh_shuf", "ol_real", "ol_shuf")} for n in NS}
    with torch.no_grad():
        for dom, seqs in data.items():
            for seq in seqs:
                KV = keys_values(torch.tensor(seq))
                for (li, h) in sorted(long_heads[dom]):
                    K, V = KV[li][0][:, h], unitv(KV[li][1][:, h])
                    h2 = (h + 1 + rng.integers(g.H - 1)) % g.H
                    Voh = unitv(KV[li][1][:, h2])
                    lj = (li + 1 + rng.integers(g.NL - 1)) % g.NL                   # a different layer
                    Vol = unitv(KV[lj][1][:, rng.integers(g.H)])
                    for n in NS:
                        for end in ENDS:
                            s0 = end - n + 1
                            Kw = K[s0:end + 1]; T = err_op(Kw)
                            for tag, Vs in (("own", V), ("oh", Voh), ("ol", Vol)):
                                Vw = Vs[s0:end + 1]
                                res[n][tag + "_real"].append(rms(T, Vw))
                                res[n][tag + "_shuf"].append(np.mean([rms(T, Vw[rng.permutation(n)]) for _ in range(4)]))
                print(dom, "seq done", flush=True)
    print("\nRMS error with head h's keys and values from: own head | another head, same layer | a head of a different layer (same positions)")
    print(f"{'n':>3} | {'own real':>8} {'shuf':>6} {'gap':>6} | {'other-head real':>15} {'shuf':>6} {'gap':>6} {'gap/own':>8} | {'other-layer real':>16} {'shuf':>6} {'gap':>6} {'gap/own':>8}")
    for n in NS:
        m = {k: float(np.mean(v)) for k, v in res[n].items()}
        go, gh, gl = m["own_shuf"] - m["own_real"], m["oh_shuf"] - m["oh_real"], m["ol_shuf"] - m["ol_real"]
        print(f"{n:3d} | {m['own_real']:8.3f} {m['own_shuf']:6.3f} {go:6.3f} | {m['oh_real']:15.3f} {m['oh_shuf']:6.3f} {gh:6.3f} {gh/go:8.2f} | "
              f"{m['ol_real']:16.3f} {m['ol_shuf']:6.3f} {gl:6.3f} {gl/go:8.2f}")
