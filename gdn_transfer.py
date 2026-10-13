"""Does the calibrated ratio  RMS / psi_rms  (and  max_i F_i / max_i psi_i)  transfer?   ideal alpha = beta = 1, unit-normalised values.

Domains (all local, 4 sequences x 1024 tokens each): prose (APPS statements), code (APPS solutions),
docs (Python language reference, pydoc_data.topics), legal (distinct licence files in the venv), notes (this project's notes.md).
All 96 heads, n in {16, 32, 64}, 7 windows per head.  Head class from the decay horizon in results_gdn.json (prose & code agree):
long = A@256 > 0.5 in both, short = A@64 <= 0.1 in both, mid = the rest.
Saves per-window records to results_transfer.npz.
"""
import sys, json, glob, os, hashlib
sys.path.insert(0, ".")
import numpy as np
import torch
import gdn_extract as g
from gdn_kv import keys_values
from gdn_kv_tok import err_op

NS = [16, 32, 64]
ENDS = list(range(255, 1024, 128))
NSEQ = 4
unitv = lambda V: V / np.maximum(np.linalg.norm(V, axis=-1, keepdims=True), 1e-12)


def local_texts():
    import pydoc_data.topics as t
    docs = "\n\n".join(t.topics.values())
    seen, parts = set(), []
    for f in sorted(glob.glob(".venv/lib/python*/site-packages/**/LICEN[CS]E*", recursive=True) + glob.glob(".venv/lib/python*/site-packages/**/COPYING*", recursive=True)):
        if os.path.isfile(f):
            b = open(f, "rb").read(); h = hashlib.sha1(b).hexdigest()
            if h not in seen:
                seen.add(h); parts.append(b.decode("utf-8", "ignore"))
    return {"docs": docs, "legal": "\n\n".join(parts), "notes": open("notes.md").read()}


def chunks(tok, text, n_seq, T=1024):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    assert len(ids) >= n_seq * (T - 1), len(ids)
    starts = np.linspace(0, len(ids) - (T - 1), n_seq).astype(int)
    return [[tok.bos_token_id] + ids[s:s + T - 1] for s in starts]


if __name__ == "__main__":
    tok = g.AutoTokenizer.from_pretrained(g.MD)
    seqs = {k: v[:NSEQ] for k, v in g.load_sequences(tok, n_per_domain=NSEQ).items()}
    for k, txt in local_texts().items():
        seqs[k] = chunks(tok, txt, NSEQ)
    per = json.load(open("results_gdn.json"))["per_head"]
    cls = {}
    for l in range(g.NL):
        for h in range(g.H):
            a, b = per[f"prose|{l}|{h}"], per[f"code|{l}|{h}"]
            cls[(l, h)] = "long" if (a["p256_A"] > .5 and b["p256_A"] > .5) else ("short" if (a["p64_A"] <= .1 and b["p64_A"] <= .1) else "mid")
    print("head classes:", {c: sum(v == c for v in cls.values()) for c in ("long", "mid", "short")}, flush=True)
    rec = []
    with torch.no_grad():
        for di, (dom, ss) in enumerate(seqs.items()):
            for si, seq in enumerate(ss):
                KV = keys_values(torch.tensor(seq))
                for li in range(g.NL):
                    for h in range(g.H):
                        K, V = KV[li][0][:, h], unitv(KV[li][1][:, h])
                        for n in NS:
                            for end in ENDS:
                                Kw, Vw = K[end - n + 1:end + 1], V[end - n + 1:end + 1]
                                T = err_op(Kw)
                                TK = T @ Kw
                                E = T @ Vw
                                rec.append((di, si, li, h, ["long", "mid", "short"].index(cls[(li, h)]), n,
                                            np.linalg.norm(E) / np.sqrt(n), np.linalg.norm(TK) / np.sqrt(n),
                                            np.linalg.norm(E, axis=1).max(), np.linalg.norm(TK, axis=1).max()))
                print(dom, si, "done", flush=True)
    A = np.array(rec)
    np.savez("results_transfer.npz", A=A, domains=np.array(list(seqs)))
    print("saved", A.shape)
