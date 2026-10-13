"""Validate our RWKV-7 re-implementation (rwkv7_flat.py) against the author's independent NumPy reference (BlinkDL/RWKV-LM, RWKV-v7/rwkv_v7_numpy.py).

The reference functions (layer_norm, group_norm, time_mixing, channel_mixing, RWKV7) are taken from the downloaded file UNMODIFIED: only the
top-level statements before the "# Verification" marker are executed (the torch import is replaced by a stub; the file's own verification block,
which needs the original .pth checkpoint, `pip install rwkv` and hard-coded paths, is not run). The weights are the cached fla-hub/rwkv7-0.1B-g1
tensors (converted to the reference's parameter layout below), so no further download is needed.

Usage: python -I validate_rwkv7_reference.py <path to rwkv_v7_numpy.py> [n_tokens]
Writes validate_rwkv7_reference.json.
"""
import sys, os, ast, json, time
import numpy as np

REF = sys.argv[1]; NTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 512
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.argv = [sys.argv[0], "check"]            # rwkv7_flat reads sys.argv[1] at import time
import rwkv7_flat as R

NL, D, H, N = R.NL, R.D, R.H, R.N


def load_reference(dtype):
    src = open(REF).read(); tree = ast.parse(src); keep = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "torch": continue          # not needed: weights are supplied below
        keep.append(node)
        if isinstance(node, ast.FunctionDef) and node.name == "RWKV7": break               # everything after this is the file's own verification block
    ns = {"N_LAYER": NL, "N_EMBD": D, "HEAD_SIZE": N, "N_HEAD": H}
    exec(compile(ast.Module(body=keep, type_ignores=[]), REF, "exec"), ns)
    return ns


def build_params(W, dtype):
    """fla names -> the parameter lists in the order of the reference's `blocks.i.att` / `.ffn` keys (RWKV-LM checkpoint order)."""
    f = lambda a: np.asarray(a, dtype)
    tab = {"emb": [f(W["model.embeddings.weight"])], "head": [f(W["lm_head.weight"])], "ln_out": [f(W["model.norm.weight"]), f(W["model.norm.bias"])],
           "blocks.0.ln0": [f(W["model.layers.0.pre_norm.weight"]), f(W["model.layers.0.pre_norm.bias"])]}
    for i in range(NL):
        g = lambda n: W[f"model.layers.{i}.{n}"]
        a = lambda n: f(g("attn." + n))
        att = [f(g("attn.x_" + k).squeeze()) for k in "rwkvag"]                            # mr, mw, mk, mv, ma, mg
        att += [a("w_lora.lora.2.bias"), a("r_k"), a("w_lora.lora.0.weight").T, a("w_lora.lora.2.weight").T,
                a("a_lora.lora.0.weight").T, a("a_lora.lora.2.weight").T, a("a_lora.lora.2.bias"), a("g_lora.lora.0.weight").T, a("g_lora.lora.2.weight").T]
        if i > 0: att += [a("v_lora.lora.2.weight").T, a("v_lora.lora.0.weight").T, a("v_lora.lora.2.bias")]   # Wv2, Wv1, v_bias
        att += [a("k_k"), a("k_a"), a("r_proj.weight"), a("k_proj.weight"), a("v_proj.weight"), a("o_proj.weight"), a("g_norm.weight"), a("g_norm.bias")]
        tab[f"blocks.{i}.att"] = att
        tab[f"blocks.{i}.ln1"] = [f(g("attn_norm.weight")), f(g("attn_norm.bias"))]
        tab[f"blocks.{i}.ln2"] = [f(g("ffn_norm.weight")), f(g("ffn_norm.bias"))]
        tab[f"blocks.{i}.ffn"] = [f(g("ffn.x_k")), f(g("ffn.key.weight")), f(g("ffn.value.weight"))]
    return lambda prefix: tab[prefix]


def run_reference(ns, params, ids, dtype):
    state = (np.zeros((NL, 2, D), dtype), np.zeros((NL, H, N, N), dtype)); out = []
    for t in ids:
        lg, state = ns["RWKV7"](params, int(t), state); out.append(lg)
    return np.array(out)


def logsoftmax(x):
    x = x - x.max(-1, keepdims=True); return x - np.log(np.exp(x).sum(-1, keepdims=True))


def compare(ours, ref, ids):
    ours = ours.astype(np.float64); ref = ref.astype(np.float64); y = np.asarray(ids)[1:]
    lo, lr = logsoftmax(ours), logsoftmax(ref)
    kl = (np.exp(lr) * (lr - lo)).sum(-1)
    return dict(max_abs_dlogit=float(np.abs(ours - ref).max()), rel_to_logit_std=float(np.abs(ours - ref).max() / ref.std()), mean_abs_logit=float(np.abs(ref).mean()),
                max_kl=float(kl.max()), top1_agree=float((ours.argmax(-1) == ref.argmax(-1)).mean()),
                ppl_ours=float(np.exp(-lo[:-1][np.arange(len(y)), y].mean())),
                ppl_ref=float(np.exp(-lr[:-1][np.arange(len(y)), y].mean())))


def main():
    trie = R.load_tokenizer(); M = R.Model(); tx = R.texts(); res = {}
    ns = load_reference(None); W = {k: v for k, v in M.W.items()}
    P64, P32 = build_params(W, np.float64), build_params(W, np.float32)
    for dom in ("prose", "code", "notes"):
        ids = np.array(R.encode(trie, tx[dom])[:NTOK]); t0 = time.time()
        h, _ = M.forward(ids); ours = h @ W["lm_head.weight"].T
        ref64 = run_reference(ns, P64, ids, np.float64); ref32 = run_reference(ns, P32, ids, np.float32)
        res[dom] = dict(tokens=len(ids), float64=compare(ours, ref64, ids), float32=compare(ours, ref32, ids))
        print(dom, len(ids), "tokens", f"[{time.time()-t0:.0f}s]", json.dumps(res[dom]), flush=True)
    res["reference_file"] = os.path.basename(REF)
    json.dump(res, open(os.path.join(HERE, "validate_rwkv7_reference.json"), "w"), indent=1)


main()
