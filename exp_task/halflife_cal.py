"""Median per-token half-life of each head's decay gate on calibration prompts (j even), no outcomes used.  Torch only."""
import os, sys, random, time
HERE = "."; sys.path.insert(0, HERE); sys.path.insert(0, HERE + "/exp_task")
import numpy as np, torch
import gdn_extract as G
G.PROBES, G.DELTAS = [], []; torch.set_num_threads(4)
from recall_lib import build_vocab, make_prompt
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(G.MD); vocab = build_vocab(tok); bos = tok.bos_token_id; name = os.path.basename(G.MD.rstrip("/"))
LOADS = [4, 8, 16, 24]; hl = {(li, h): [] for li in range(G.NL) for h in range(G.H)}; t0 = time.time()
with torch.no_grad():
    for n in LOADS:
        for j in range(0, 20, 2):                                    # calibration prompts only (j even)
            p = make_prompt(random.Random(10_000 * n + j), vocab, n, bos); _, col = G.forward(torch.tensor(p["ids"]), collect_layers=set(range(G.NL)))
            for li in range(G.NL):
                al = torch.from_numpy(col[li]["alpha"][: p["qpos"]]).float()
                for h in range(G.H): hl[(li, h)].append((np.log(2) / np.maximum(-torch.log(al[:, h]).numpy(), 1e-9)))
med = np.array([[np.median(np.concatenate(hl[(li, h)])) for h in range(G.H)] for li in range(G.NL)])
np.save(f"{HERE}/exp_task/out/{name}_halflife_cal.npy", med); print(name, "median half-life >=256:", int((med >= 256).sum()), "of", med.size, " >=64:", int((med >= 64).sum()), " range", med.min(), med.max(), f"[{time.time()-t0:.0f}s]")
