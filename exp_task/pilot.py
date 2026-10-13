"""Pilot: accuracy of the model on the recall task versus the number of facts (no certificate computation)."""
import os, sys, time, json, random
sys.path.insert(0, "."); sys.path.insert(0, "./exp_task")
import torch
import gdn_extract as G
from recall_lib import build_vocab, make_prompt
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(G.MD); vocab = build_vocab(tok); bos = tok.bos_token_id
print("model", G.MD, "names", len(vocab[0]), "words", len(vocab[1]), "bos", bos, flush=True)
rng = random.Random(0); res = {}
t0 = time.time()
with torch.no_grad():
    for n in [2, 4, 8, 16, 32]:
        ok, lp, fc = [], [], []
        for _ in range(int(sys.argv[1]) if len(sys.argv) > 1 else 24):
            p = make_prompt(rng, vocab, n, bos); logits, _ = G.forward(torch.tensor(p["ids"]))
            last = logits[-1]; cand = [pp for pp in p['cands']]; fc.append(int(max(cand, key=lambda t: float(last[t])) == p['target'])); ok.append(int(last.argmax()) == p["target"]); lp.append(float(torch.log_softmax(last, -1)[p["target"]]))
        res[n] = (sum(ok) / len(ok), sum(lp) / len(lp)); print(f"n={n:3d} top1-vocab={res[n][0]:.2f}  forced-choice={sum(fc)/len(fc):.2f} (chance {1/n:.2f})  mean logp(true)={res[n][1]:.2f} [{time.time()-t0:.0f}s]", flush=True)
