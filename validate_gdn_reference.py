"""Validate the PyTorch re-implementation of Gated DeltaNet (gdn_extract.py) against an independent reference: the GatedDeltaNet layer of
HuggingFace transformers (Qwen3-Next family; torch chunked gated-delta-rule kernel), with the checkpoint's weights mapped into that layer.
  A. layer level: identical hidden input, compare outputs of every checked layer
  B. model level: swap the reference layer into the full forward pass, compare logits (max |diff|, KL, top-1 agreement) and perplexity
Run:  GDN_MD=model_gdn340m ./.venv/bin/python -I validate_gdn_reference.py   (no checkpoint code is executed; transformers is the only library)"""
import sys, os, json
sys.path.insert(0, ".")
import numpy as np, torch, torch.nn.functional as F
import gdn_extract as g
from transformers import AutoTokenizer
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextGatedDeltaNet

torch.set_num_threads(10); H, DK, DV, NL = g.H, g.DK, g.DV, g.NL
cfg = Qwen3NextConfig(hidden_size=g.cfg["hidden_size"], num_hidden_layers=NL, linear_num_value_heads=H, linear_num_key_heads=H, linear_key_head_dim=DK, linear_value_head_dim=DV,
                      linear_conv_kernel_dim=g.cfg["conv_size"], rms_norm_eps=g.EPS, hidden_act="silu", layer_types=["linear_attention"] * NL)
W = g.W


def ref_layer(i):
    p = lambda n: W[f"model.layers.{i}.attn.{n}"]; L = Qwen3NextGatedDeltaNet(cfg, i).float().eval(); hid = g.cfg["hidden_size"]
    Wq, Wk, Wv, Wz = p("q_proj.weight"), p("k_proj.weight"), p("v_proj.weight"), p("g_proj.weight"); Wb, Wa = p("b_proj.weight"), p("a_proj.weight")
    qkvz = torch.cat([torch.cat([Wq[h * DK:(h + 1) * DK], Wk[h * DK:(h + 1) * DK], Wv[h * DV:(h + 1) * DV], Wz[h * DV:(h + 1) * DV]], 0) for h in range(H)], 0)
    ba = torch.cat([torch.stack([Wb[h], Wa[h]], 0) for h in range(H)], 0)
    ident = torch.zeros(L.conv1d.weight.shape); ident[:, :, -1] = 1.0       # checkpoints without a short convolution (use_short_conv = false): identity kernel, so the reference computes SiLU(projection)
    with torch.no_grad():
        L.in_proj_qkvz.weight.copy_(qkvz); L.in_proj_ba.weight.copy_(ba); L.conv1d.weight.copy_(torch.cat([p("q_conv1d.weight"), p("k_conv1d.weight"), p("v_conv1d.weight")], 0) if f"model.layers.{i}.attn.q_conv1d.weight" in W else ident)
        L.A_log.copy_(p("A_log")); L.dt_bias.copy_(p("dt_bias")); L.norm.weight.copy_(p("o_norm.weight")); L.out_proj.weight.copy_(p("o_proj.weight"))
    return L


REF = {}
def ref_attn(x, i):
    if i not in REF: REF[i] = ref_layer(i)
    with torch.no_grad(): return REF[i](x[None])[0]


def forward(tokens, use_ref, record=None):
    h = W["model.embeddings.weight"][tokens]
    for i in range(NL):
        x = g.rms(h, W[f"model.layers.{i}.attn_norm.weight"])
        a = ref_attn(x, i) if use_ref else g.gdn(x, i, False)[0]
        if record is not None and i in record: record[i] = (x, a)
        h = h + a; h = h + g.mlp(g.rms(h, W[f"model.layers.{i}.mlp_norm.weight"]), i)
    h = g.rms(h, W["model.norm.weight"]); return h @ W.get("lm_head.weight", W["model.embeddings.weight"]).T


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(g.MD); seqs = g.load_sequences(tok, n_per_domain=2); out = dict(model=g.MD, layers={}, model_level=[])
    S = [("prose", seqs["prose"][0]), ("code", seqs["code"][0])]
    for name, ids in S:
        t = torch.tensor(ids)
        with torch.no_grad():
            rec = {i: None for i in (0, NL // 4, NL // 2, 3 * NL // 4, NL - 1)}
            lm = forward(t, False, rec)
            lr = forward(t, True)
        for i, (x, a) in rec.items():
            r = ref_attn(x, i); out["layers"][f"{name}_L{i}"] = float((a - r).abs().max() / r.abs().max())
        pm, pr = F.log_softmax(lm, -1), F.log_softmax(lr, -1); kl = (pm.exp() * (pm - pr)).sum(-1)
        nll = lambda lg: float(F.cross_entropy(lg[:-1], t[1:]))
        out["model_level"].append(dict(seq=name, max_abs_logit_diff=float((lm - lr).abs().max()), mean_abs_logit_diff=float((lm - lr).abs().mean()), logit_scale=float(lm.abs().mean()), mean_kl=float(kl.mean()), max_kl=float(kl.max()),
                                      top1_agree=float((lm.argmax(-1) == lr.argmax(-1)).float().mean()), ppl_mine=float(np.exp(nll(lm))), ppl_ref=float(np.exp(nll(lr)))))
        print(name, out["model_level"][-1], flush=True)
    json.dump(out, open(f"validate_gdn_{os.path.basename(g.MD)}.json", "w"), indent=1); print(json.dumps(out["layers"], indent=1))
