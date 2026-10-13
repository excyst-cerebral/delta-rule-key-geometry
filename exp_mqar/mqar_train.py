"""Associative recall after distractors (MQAR-style) with a small delta-rule model, plus task-agnostic regularisers motivated by the theory.

Sequence: n (key, value) pairs, then L filler tokens, then the n keys again in random order; the target at each query position is its value.
Forgetting is measured as recall accuracy vs (n pairs, distractor length L).
Model: embedding -> 2 x [ gated delta-rule layer (short conv, L2-normalised keys, beta, decay) + SwiGLU MLP ] -> logits. Torch only.
Regularisers (computed on the first W = 16 positions, always inside the pair region; unit-norm rows):
  dec   : mean squared off-diagonal key cosine            (key decorrelation: raises the keys' effective rank)
  match : mean squared off-diagonal (H - G) entries        (value Gram matched to the key Gram: pushes the relative gain to 1)
  rank  : relu(r_eff(H) - r_eff(G))                         (values no higher-rank than keys)
"""
import argparse, json, math, os, time
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch, torch.nn as nn, torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--reg", default="none"); p.add_argument("--lam", type=float, default=0.0)
p.add_argument("--seed", type=int, default=0); p.add_argument("--steps", type=int, default=3000)
p.add_argument("--d", type=int, default=64); p.add_argument("--heads", type=int, default=2); p.add_argument("--dk", type=int, default=32)
p.add_argument("--batch", type=int, default=64); p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--nmax", type=int, default=32); p.add_argument("--Lmax", type=int, default=96)
p.add_argument("--threads", type=int, default=2); p.add_argument("--out", default="")
p.add_argument("--timing", action="store_true")
a = p.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
import transformers.models.qwen3_5.modeling_qwen3_5 as _mq   # reference chunked delta rule (checked equal to the naive loop: fwd 7e-8, grad 3e-7)
chunk_rule = _mq.torch_chunk_gated_delta_rule
NK, NV, NF = 64, 64, 32
VOCAB = NK + NV + NF


def make_batch(B, n, L, gen):
    keys = torch.stack([torch.randperm(NK, generator=gen)[:n] for _ in range(B)])                      # (B,n)
    vals = NK + torch.randint(0, NV, (B, n), generator=gen)
    pairs = torch.stack([keys, vals], -1).reshape(B, 2 * n)
    fill = NK + NV + torch.randint(0, NF, (B, L), generator=gen)
    perm = torch.stack([torch.randperm(n, generator=gen) for _ in range(B)])
    qk = torch.gather(keys, 1, perm); qv = torch.gather(vals, 1, perm)
    x = torch.cat([pairs, fill, qk], 1)
    y = torch.full_like(x, -100); y[:, 2 * n + L:] = qv
    return x, y


class RMSNorm(nn.Module):
    def __init__(s, d, eps=1e-6):
        super().__init__(); s.w = nn.Parameter(torch.ones(d)); s.eps = eps
    def forward(s, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + s.eps) * s.w


class DeltaLayer(nn.Module):
    def __init__(s, d, H, dk, ks=4):
        super().__init__(); s.H, s.dk, s.dv = H, dk, dk
        s.q, s.k, s.v = (nn.Linear(d, H * dk, bias=False) for _ in range(3))
        s.ks = ks; s.cq, s.ck, s.cv = (nn.Parameter(torch.randn(H * dk, ks) / math.sqrt(ks)) for _ in range(3))   # depthwise causal conv weights (C, ks)
        s.a, s.b = nn.Linear(d, H, bias=False), nn.Linear(d, H, bias=False)
        s.A_log = nn.Parameter(torch.log(torch.empty(H).uniform_(1, 16)))
        dt = torch.exp(torch.rand(H) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        s.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        s.g = nn.Linear(d, H * dk, bias=False); s.on = RMSNorm(dk); s.o = nn.Linear(H * dk, d, bias=False)
        s.cache = None
    def forward(s, x):
        B, T, _ = x.shape; H, dk = s.H, s.dk
        def cv(w, u):                                              # causal depthwise conv (shift-and-sum; equals Conv1d with left padding) + SiLU
            up = F.pad(u, (0, 0, s.ks - 1, 0))
            return F.silu(sum(up[:, j:j + T] * w[:, j] for j in range(s.ks)))
        q = cv(s.cq, s.q(x)).view(B, T, H, dk); k = cv(s.ck, s.k(x)).view(B, T, H, dk); v = cv(s.cv, s.v(x)).view(B, T, H, dk)
        q = F.normalize(q, dim=-1); k = F.normalize(k, dim=-1)
        beta = torch.sigmoid(s.b(x)); logalpha = -torch.exp(s.A_log) * F.softplus(s.a(x) + s.dt_bias); alpha = torch.exp(logalpha)
        o, _ = chunk_rule(q, k, v, g=logalpha, beta=beta, chunk_size=16, output_final_state=False, use_qk_l2norm_in_kernel=False)   # applies dk^-0.5 to q
        s.cache = (k, v, beta, alpha)
        return s.o((s.on(o) * F.silu(s.g(x)).view(B, T, H, dk)).reshape(B, T, H * dk))


class Block(nn.Module):
    def __init__(s, d, H, dk):
        super().__init__(); s.n1, s.n2 = RMSNorm(d), RMSNorm(d); s.attn = DeltaLayer(d, H, dk)
        s.u, s.w, s.dn = nn.Linear(d, 2 * d, bias=False), nn.Linear(d, 2 * d, bias=False), nn.Linear(2 * d, d, bias=False)
    def forward(s, x):
        x = x + s.attn(s.n1(x)); h = s.n2(x)
        return x + s.dn(F.silu(s.u(h)) * s.w(h))


class Net(nn.Module):
    def __init__(s, d, H, dk, nl=2):
        super().__init__(); s.emb = nn.Embedding(VOCAB, d); s.blocks = nn.ModuleList(Block(d, H, dk) for _ in range(nl)); s.nf = RMSNorm(d)
        s.head = nn.Linear(d, VOCAB, bias=False)
    def forward(s, x):
        h = s.emb(x)
        for b in s.blocks: h = b(h)
        return s.head(s.nf(h))


def reg_loss(model, kind, W=16):
    if kind == "none": return torch.zeros(())
    tot = 0.0
    W = min(W, model.blocks[0].attn.cache[0].shape[1])       # sequences can be shorter than W
    for blk in model.blocks:
        k, v, _, _ = blk.attn.cache
        K = k[:, :W].permute(0, 2, 1, 3); V = F.normalize(v[:, :W], dim=-1).permute(0, 2, 1, 3)       # (B,H,W,d)
        G, Hm = K @ K.transpose(-1, -2), V @ V.transpose(-1, -2)
        off = 1 - torch.eye(W)
        if kind == "dec": r = ((G * off) ** 2).sum((-1, -2)) / (W * (W - 1))
        elif kind == "match": r = (((Hm - G) * off) ** 2).sum((-1, -2)) / (W * (W - 1))
        elif kind == "rank":
            reff = lambda M: (M.diagonal(dim1=-2, dim2=-1).sum(-1) ** 2) / (M ** 2).sum((-1, -2))
            r = F.relu(reff(Hm) - reff(G))
        else: raise ValueError(kind)
        tot = tot + r.mean()
    return tot


@torch.no_grad()
def evaluate(model, ns, Ls, gen, B=128):
    model.eval(); res = {}
    for n in ns:
        for L in Ls:
            x, y = make_batch(B, n, L, gen); lg = model(x); m = y != -100
            res[f"{n}_{L}"] = float((lg.argmax(-1)[m] == y[m]).float().mean())
    model.train(); return res


if __name__ == "__main__":
    model = Net(a.d, a.heads, a.dk); gen = torch.Generator().manual_seed(1000 + a.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
    t0 = time.time(); hist = []
    for step in range(a.steps):
        n = int(torch.randint(4, a.nmax + 1, (1,), generator=gen)); L = int(torch.randint(0, a.Lmax + 1, (1,), generator=gen))
        x, y = make_batch(a.batch, n, L, gen)
        lg = model(x); ce = F.cross_entropy(lg.reshape(-1, VOCAB), y.reshape(-1), ignore_index=-100)
        rg = reg_loss(model, a.reg); loss = ce + a.lam * rg
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if step % 50 == 0 or step == a.steps - 1:
            hist.append((step, float(ce), float(rg))); print(f"step {step} ce {float(ce):.3f} reg {float(rg):.4f} n={n} L={L} [{time.time()-t0:.0f}s]", flush=True) if step % 250 == 0 or step == a.steps - 1 else None
        if a.timing and step == 9: print(f"{(time.time()-t0)/10:.3f} s/step"); raise SystemExit
    grid = evaluate(model, [8, 16, 24, 32], [0, 32, 96], torch.Generator().manual_seed(7))
    # save an evaluation batch's keys/values (both layers) for the post-hoc geometry analysis (numpy process)
    model.eval()
    with torch.no_grad():
        x, y = make_batch(64, 16, 32, torch.Generator().manual_seed(11)); model(x)
        cache = {f"k{i}": b.attn.cache[0].numpy() for i, b in enumerate(model.blocks)} | {f"v{i}": b.attn.cache[1].numpy() for i, b in enumerate(model.blocks)} \
              | {f"beta{i}": b.attn.cache[2].numpy() for i, b in enumerate(model.blocks)} | {f"alpha{i}": b.attn.cache[3].numpy() for i, b in enumerate(model.blocks)}
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(dict(args=vars(a), grid=grid, hist=hist, time=time.time() - t0), open(a.out + ".json", "w"))
        import numpy as np; np.savez_compressed(a.out + "_cache.npz", **cache)
    print("GRID", json.dumps({k: round(v, 3) for k, v in grid.items()}))
