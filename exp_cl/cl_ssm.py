"""Weight-space continual learning in a small SSM (Option B).  Torch only.

Tasks: m tasks with DISJOINT vocabularies (16 tokens each) and random +-1 sign maps; a length-L sequence of tokens from one task's vocabulary is
labelled 1 if the signed sum is positive.  Embeddings are FIXED random vectors (d_e = 16 < 16 m tokens), so a shared trainable SSM + readout
cannot fit every task with a linear rule: interference is real.  Learner state between tasks = parameters only (plus, for the reference methods,
exactly the extra object each method needs).

Architectures:  'lti'  : diagonal S4D-style linear time-invariant SSM (constant Delta, B, C)
                'sel'  : Mamba-style selective SSM (Delta, B, C are functions of the current input; per-channel gating)
Methods:        'ft' fine-tuning (no replay, no anchor) | 'ewc' (ANCHOR: stored parameters + Fisher) | 'replay' (REPLAY: stored examples)
                'joint' (all tasks at once; capacity reference)
"""
import argparse, json, math, os, time
import torch, torch.nn as nn, torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--arch", default="sel"); p.add_argument("--method", default="ft"); p.add_argument("--seed", type=int, default=0)
p.add_argument("--tasks", type=int, default=4); p.add_argument("--steps", type=int, default=400); p.add_argument("--L", type=int, default=15)
p.add_argument("--C", type=int, default=24); p.add_argument("--N", type=int, default=4); p.add_argument("--de", type=int, default=16)
p.add_argument("--lr", type=float, default=3e-3); p.add_argument("--ewc", type=float, default=200.0); p.add_argument("--buf", type=int, default=64)
p.add_argument("--threads", type=int, default=1); p.add_argument("--out", default="")
p.add_argument("--reg", default="none"); p.add_argument("--lam", type=float, default=0.0)
a = p.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
TOK = 16
gen = torch.Generator().manual_seed(100 + a.seed)
SIGN = [torch.randint(0, 2, (TOK,), generator=gen).float() * 2 - 1 for _ in range(a.tasks)]
E = torch.randn(a.tasks * TOK, a.de, generator=gen) / math.sqrt(a.de)                       # fixed embeddings (not trained)


def batch(t, B, g):
    tok = torch.randint(0, TOK, (B, a.L), generator=g)
    y = ((SIGN[t][tok].sum(1)) > 0).long()
    return tok + t * TOK, y


class SSM(nn.Module):
    def __init__(s, arch, C, N, de):
        super().__init__(); s.arch, s.C, s.N = arch, C, N
        s.inp = nn.Linear(de, C)
        s.A_log = nn.Parameter(torch.log(torch.rand(C, N) * 2 + 0.5))
        s.D = nn.Parameter(torch.zeros(C))
        if arch == "sel":
            s.dt = nn.Linear(C, C); s.Bp = nn.Linear(C, N); s.Cp = nn.Linear(C, N)
            nn.init.constant_(s.dt.bias, -1.0)
        else:
            s.dt_b = nn.Parameter(torch.full((C,), -1.0)); s.Bc = nn.Parameter(torch.randn(N) / math.sqrt(N)); s.Cc = nn.Parameter(torch.randn(N) / math.sqrt(N))
        s.out = nn.Linear(C, 2)
        s.cache = None
    def features(s, e):
        Bsz, L, _ = e.shape; u = s.inp(e); A = -torch.exp(s.A_log)
        if s.arch == "sel":
            dt = F.softplus(s.dt(u)); Bt = s.Bp(u); Ct = s.Cp(u)
        else:
            dt = F.softplus(s.dt_b).expand(Bsz, L, s.C); Bt = s.Bc.expand(Bsz, L, s.N); Ct = s.Cc.expand(Bsz, L, s.N)
        h = u.new_zeros(Bsz, s.C, s.N); ys = []
        for t in range(L):
            dA = torch.exp(dt[:, t, :, None] * A)
            h = dA * h + dt[:, t, :, None] * Bt[:, t, None, :] * u[:, t, :, None]
            ys.append((h * Ct[:, t, None, :]).sum(-1) + s.D * u[:, t])
        s.cache = dt
        return ys[-1]
    def forward(s, tok):
        return s.out(s.features(E[tok]))


@torch.no_grad()
def acc(model, t, g, B=1000):
    model.eval(); x, y = batch(t, B, g); r = (model(x).argmax(-1) == y).float().mean().item(); model.train(); return r


def fisher(model, t, g, n=400):
    x, y = batch(t, n, g); model.zero_grad(); out = model(x)
    # empirical Fisher on per-batch gradients of mini-batches of 1 would be costly; use 20 mini-batches of 20
    F_ = {k: torch.zeros_like(v) for k, v in model.named_parameters()}
    for i in range(0, n, 20):
        model.zero_grad(); F.cross_entropy(model(x[i:i + 20]), y[i:i + 20]).backward()
        for k, v in model.named_parameters(): F_[k] += v.grad.detach() ** 2 / (n // 20)
    model.zero_grad(); return F_


def gate_reg(model, tok, kind):
    dt = model.cache                                              # (B, L, C) gate activations of the current batch
    if kind == "sparse": return dt.sum(-1).mean()                 # L1: few channels open per token
    if kind == "decor":                                           # decorrelate the gate profiles of different tokens of the current task
        prof = torch.stack([dt[tok == tk].mean(0) if (tok == tk).any() else dt.new_zeros(dt.shape[-1]) for tk in tok.unique()])
        prof = F.normalize(prof, dim=-1); G = prof @ prof.T; n = len(G)
        return ((G - torch.eye(n)) ** 2).sum() / max(n * (n - 1), 1)
    return torch.zeros(())


def run():
    model = SSM(a.arch, a.C, a.N, a.de); opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    g = torch.Generator().manual_seed(1000 + a.seed); gt = torch.Generator().manual_seed(7)
    M = a.tasks; accm = torch.zeros(M, M)                      # accm[i, j] = accuracy on task j after training task i
    anchors, buffers = [], []
    gates = {}; init_loss = {}
    order = [None] if a.method == "joint" else list(range(M))
    for i, t in enumerate(order):
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
        for step in range(a.steps * (M if t is None else 1)):
            if t is None:
                tt = int(torch.randint(0, M, (1,), generator=g)); x, y = batch(tt, 64, g)
            else:
                x, y = batch(t, 64, g)
            loss = F.cross_entropy(model(x), y)
            if step == 0 and t is not None: init_loss[t] = float(loss)       # surprise on the first batch of the new task
            if a.reg != "none" and a.arch == "sel": loss = loss + a.lam * gate_reg(model, x, a.reg)
            if a.method == "ewc" and anchors:
                for (th, Fi) in anchors:
                    loss = loss + a.ewc * sum((Fi[k] * (v - th[k]) ** 2).sum() for k, v in model.named_parameters())
            if a.method == "replay" and buffers:
                bx = torch.cat([b[0] for b in buffers]); by = torch.cat([b[1] for b in buffers])
                idx = torch.randint(0, len(bx), (64,), generator=g); loss = loss + F.cross_entropy(model(bx[idx]), by[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        if t is not None:
            if a.method == "ewc": anchors.append(({k: v.detach().clone() for k, v in model.named_parameters()}, fisher(model, t, g)))
            if a.method == "replay": buffers.append(batch(t, a.buf, g))
            if a.arch == "sel":                                           # mean gate (Delta) per channel on each task's data
                for j in range(M):
                    with torch.no_grad(): x, _ = batch(j, 500, gt); model(x); gates[(i, j)] = model.cache.mean((0, 1)).tolist()
        for j in range(M): accm[0 if t is None else i, j] = acc(model, j, gt)
    return accm, gates, init_loss


if __name__ == "__main__":
    t0 = time.time(); accm, gates, init_loss = run(); M = a.tasks
    if a.method == "joint": final = accm[0]; forget = 0.0
    else:
        final = accm[-1]; forget = float(sum(accm[:M - 1, j].max() - accm[-1, j] for j in range(M - 1)) / (M - 1))
    res = dict(args=vars(a), acc_matrix=accm.tolist(), final_avg=float(final.mean()), forgetting=forget, gates={f"{k[0]}_{k[1]}": v for k, v in gates.items()}, init_loss={str(k): v for k, v in init_loss.items()}, time=time.time() - t0)
    print(json.dumps(dict(arch=a.arch, method=a.method, reg=a.reg, lam=a.lam, seed=a.seed, final_avg=round(res["final_avg"], 3), forgetting=round(forget, 3), time=round(res["time"]))))
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(res, open(a.out + ".json", "w"))
