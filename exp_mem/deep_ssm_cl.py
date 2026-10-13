"""Deep diagonal SSM, continual learning of its WEIGHTS, memory (bits) vs forgetting vs plasticity.  Torch only.
Tasks: length-48 scalar sequences, noise N(0, 0.1^2); a +-1 pulse sits `d_t` steps before the end; label = sign(pulse) * s_t (task polarity s_t random).
A linear time-invariant model reads the sign through its impulse response K(d); tasks conflict through the shared kernel (time-scale structure).
Model: Linear(1->C) -> SSM -> SiLU -> Linear(C->C) -> SSM -> last step -> Linear(C->2).  SSM: real diagonal poles lambda=sigmoid(raw) (C x N), per-channel B, C, D.
Learners (plain SGD, no momentum):
  ft      fine-tuning (pure)
  ogd     stores a basis of past per-sample logit-gradient directions (energy rho, q bits/entry) and projects gradients off it   [ANCHOR: stored subspace]
  ewc     stores theta* and diagonal Fisher per task                                                                        [ANCHOR]
  replay  stores k examples per task                                                                                        [REPLAY]
  gpm     layerwise: stores, per linear layer, a basis of old input-activation subspace (energy rho, q bits) and projects the layer's
          weight(+bias) gradient off it; recurrence params: --ssm free | freeze (after task 0) | ogd (gradient subspace on those coordinates)  [ANCHOR]
"""
import argparse, json, math, os, time
import torch, torch.nn as nn, torch.nn.functional as F

p_ = argparse.ArgumentParser()
p_.add_argument("--method", default="ft"); p_.add_argument("--seed", type=int, default=0); p_.add_argument("--rho", type=float, default=0.95)
p_.add_argument("--q", type=int, default=32); p_.add_argument("--k", type=int, default=16); p_.add_argument("--ewc", type=float, default=100.0)
p_.add_argument("--steps", type=int, default=600); p_.add_argument("--lr", type=float, default=0.03); p_.add_argument("--C", type=int, default=12); p_.add_argument("--N", type=int, default=4)
p_.add_argument("--drift", action="store_true"); p_.add_argument("--ssm", default="free"); p_.add_argument("--probe", action="store_true"); p_.add_argument("--ns", type=int, default=256); p_.add_argument("--tasks", type=int, default=5); p_.add_argument("--T", type=int, default=48); p_.add_argument("--threads", type=int, default=1); p_.add_argument("--out", default="")
a = p_.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
DELAYS = [2, 5, 9, 14, 20][:a.tasks]
gs = torch.Generator().manual_seed(500 + a.seed)
POL = torch.tensor([1., -1., 1., -1., 1.][:a.tasks])[torch.randperm(a.tasks, generator=gs)] if a.tasks > 1 else torch.ones(1)


def batch(t, B, g):
    u = 0.1 * torch.randn(B, a.T, generator=g); s = torch.randint(0, 2, (B,), generator=g).float() * 2 - 1
    u[:, a.T - 1 - DELAYS[t]] += 1.5 * s
    return u, ((s * POL[t]) > 0).long()


class SSM(nn.Module):
    def __init__(s, C, N):
        super().__init__(); lam = torch.linspace(0.3, 0.98, C * N)[torch.randperm(C * N)].view(C, N)
        s.raw = nn.Parameter(torch.logit(lam)); s.B = nn.Parameter(torch.randn(C, N) / math.sqrt(N)); s.Cm = nn.Parameter(torch.randn(C, N) / math.sqrt(N)); s.D = nn.Parameter(torch.zeros(C))
    def forward(s, u):                                   # u: (B, T, C)
        lam = torch.sigmoid(s.raw); h = u.new_zeros(u.shape[0], *s.raw.shape); ys = []
        for t in range(u.shape[1]):
            h = lam * h + s.B * u[:, t, :, None]; ys.append((h * s.Cm).sum(-1) + s.D * u[:, t])
        return torch.stack(ys, 1)


class Net(nn.Module):
    def __init__(s, C, N):
        super().__init__(); s.l1 = nn.Linear(1, C); s.s1 = SSM(C, N); s.mix = nn.Linear(C, C); s.s2 = SSM(C, N); s.head = nn.Linear(C, 2)
    def forward(s, x):
        h = F.silu(s.s1(s.l1(x[..., None]))); h = s.s2(s.mix(h))[:, -1]; return s.head(h)


def flat(model): return torch.cat([p.reshape(-1) for p in model.parameters()])
def flat_grad(model): return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in model.parameters()])
def set_grad(model, g):
    i = 0
    for p in model.parameters(): n = p.numel(); p.grad = g[i:i + n].view_as(p).clone(); i += n


@torch.no_grad()
def acc(model, t, g, B=1000):
    x, y = batch(t, B, g); return (model(x).argmax(-1) == y).float().mean().item()


def sample_grads(model, t, g, n=96):
    x, y = batch(t, n, g); G = []
    for i in range(n):
        model.zero_grad(); out = model(x[i:i + 1]); (out[0, 1] - out[0, 0]).backward(); G.append(flat_grad(model).clone())
    model.zero_grad(); return torch.stack(G)


LIN = ("l1", "mix", "head")


@torch.no_grad()
def collect_acts(model, t, g, n):
    x, _ = batch(t, n, g); store = {}
    hs = [getattr(model, nm).register_forward_pre_hook(lambda m, inp, nm=nm: store.__setitem__(nm, inp[0].reshape(-1, inp[0].shape[-1]).clone())) for nm in LIN]
    model(x)
    for h in hs: h.remove()
    return {nm: torch.cat([A, A.new_ones(A.shape[0], 1)], 1) for nm, A in store.items()}


def project_linear(model, bases):
    for nm in LIN:
        B = bases[nm]
        if B.shape[1] == 0: continue
        lay = getattr(model, nm); gw = torch.cat([lay.weight.grad, lay.bias.grad[:, None]], 1); gw = gw - gw @ B @ B.T
        lay.weight.grad = gw[:, :-1].clone(); lay.bias.grad = gw[:, -1].clone()


def gram_schmidt(Bcols, new):
    out = []
    for v in new.T:
        for _ in range(2):
            if Bcols.shape[1]: v = v - Bcols @ (Bcols.T @ v)
            for w in out: v = v - w * (w @ v)
        n = v.norm()
        if n > 1e-6: out.append(v / n)
    return torch.stack(out, 1) if out else Bcols.new_zeros(Bcols.shape[0], 0)


def quant(Bm, q):
    if q >= 32: return Bm
    lv = 2 ** (q - 1) - 1; return torch.round(Bm.clamp(-1, 1) * lv) / lv


def run():
    model = Net(a.C, a.N); p = sum(x.numel() for x in model.parameters()); g = torch.Generator().manual_seed(1000 + a.seed); gt = torch.Generator().manual_seed(7)
    gp = torch.Generator().manual_seed(99); probes = [batch(j, 200, gp)[0] for j in range(a.tasks)]; ref = {}; pchg = []
    raw = {}; lb = {nm: torch.zeros(getattr(model, nm).weight.shape[1] + 1, 0) for nm in LIN}; ssm_idx = torch.cat([torch.full((q_.numel(),), float(any(q_ is r_ for sm in (model.s1, model.s2) for r_ in sm.parameters()))) for q_ in model.parameters()]).bool(); sbasis = torch.zeros(int(ssm_idx.sum()), 0); lranks = []; M = a.tasks; accm = torch.zeros(M, M); basis = torch.zeros(p, 0); bits = 0; anchors = []; buf = []; ranks = []
    for t in range(M):
        for _ in range(a.steps):
            x, y = batch(t, 64, g); model.zero_grad(); loss = F.cross_entropy(model(x), y)
            if a.method == "replay" and buf:
                bx = torch.cat([b[0] for b in buf]); by = torch.cat([b[1] for b in buf]); idx = torch.randint(0, len(bx), (64,), generator=g); loss = loss + F.cross_entropy(model(bx[idx]), by[idx])
            if a.method == "ewc" and anchors:
                th = flat(model)
                for (ts, Fi) in anchors: loss = loss + a.ewc * (Fi * (th - ts) ** 2).sum()
            loss.backward(); gr = flat_grad(model)
            gn = gr.norm(); gr = gr * min(1.0, 5.0 / (float(gn) + 1e-12))                      # gradient clipping (before projection; projection only shrinks)
            if a.method == "ogd" and basis.shape[1]: gr = gr - basis @ (basis.T @ gr)
            set_grad(model, gr)
            if a.ssm == "fixed" and a.method != "gpm":
                gg = flat_grad(model); gg[ssm_idx] = 0; set_grad(model, gg)
            if a.method == "gpm":
                project_linear(model, lb)
                if a.ssm == "fixed":
                    gg = flat_grad(model); gg[ssm_idx] = 0; set_grad(model, gg)
                if a.ssm not in ("free", "fixed") and t > 0:
                    gg = flat_grad(model)
                    if a.ssm == "freeze": gg[ssm_idx] = 0
                    elif sbasis.shape[1]: gg[ssm_idx] = gg[ssm_idx] - sbasis @ (sbasis.T @ gg[ssm_idx])
                    set_grad(model, gg)
            with torch.no_grad():
                for prm in model.parameters(): prm -= a.lr * prm.grad
        if a.method == "ogd":
            G = sample_grads(model, t, g); R = G - G @ basis @ basis.T if basis.shape[1] else G
            U, S, Vh = torch.linalg.svd(R, full_matrices=False); e = torch.cumsum(S ** 2, 0) / (S ** 2).sum(); r = int((e < a.rho).sum()) + 1
            new = gram_schmidt(basis, quant(Vh[:r].T, a.q)); basis = torch.cat([basis, new], 1); bits += new.shape[1] * p * a.q; ranks.append(new.shape[1])
            if a.drift:
                Ur, Sr, Vr = torch.linalg.svd(G, full_matrices=False); er = torch.cumsum(Sr ** 2, 0) / (Sr ** 2).sum(); rr = int((er < a.rho).sum()) + 1; raw[t] = (Vr[:rr].T.clone(), rr)
        if a.method == "gpm":
            acts = collect_acts(model, t, g, a.ns); rr_ = []
            for nm in LIN:
                A = acts[nm]; B = lb[nm]; tot = float((A ** 2).sum()); old = float(((A @ B) ** 2).sum()) if B.shape[1] else 0.0
                R = A - A @ B @ B.T if B.shape[1] else A
                U, S, Vh = torch.linalg.svd(R, full_matrices=False); cum = (old + torch.cumsum(S ** 2, 0)) / tot; r = int((cum < a.rho).sum()) + 1
                new = gram_schmidt(B, quant(Vh[:r].T, a.q)); lb[nm] = torch.cat([B, new], 1); bits += new.shape[1] * lb[nm].shape[0] * a.q; rr_.append(new.shape[1])
            lranks.append(rr_)
            if a.ssm == "ogd":
                G = sample_grads(model, t, g)[:, ssm_idx]; R = G - G @ sbasis @ sbasis.T if sbasis.shape[1] else G
                U, S, Vh = torch.linalg.svd(R, full_matrices=False); e = torch.cumsum(S ** 2, 0) / (S ** 2).sum(); r = int((e < a.rho).sum()) + 1
                new = gram_schmidt(sbasis, quant(Vh[:r].T, a.q)); sbasis = torch.cat([sbasis, new], 1); bits += new.shape[1] * sbasis.shape[0] * a.q
        if a.method == "ewc":
            G = sample_grads(model, t, g); anchors.append((flat(model).detach().clone(), (G ** 2).mean(0))); bits += 2 * p * 32
        if a.method == "replay":
            buf.append(batch(t, a.k, g)); bits += a.k * (a.T * 32 + 1)
        for j in range(M): accm[t, j] = acc(model, j, gt)
        if a.probe:
            with torch.no_grad():
                for j in range(t):
                    d = (model(probes[j]) - ref[j]).abs(); pchg.append(dict(learned=j, now=t, max=float(d.max()), mean=float(d.mean()), scale=float(ref[j].abs().mean())))
                ref[t] = model(probes[t]).clone()
    forget = float(sum(accm[:M - 1, j].max() - accm[-1, j] for j in range(M - 1)) / (M - 1))
    drift = []
    if a.drift:
        for t in range(M - 1):
            Gf = sample_grads(model, t, g); Uf = torch.linalg.svd(Gf, full_matrices=False)[2][:raw[t][1]].T; sv = torch.linalg.svdvals(raw[t][0].T @ Uf)
            drift.append(dict(task=t, r=raw[t][1], mean_cos2=float((sv ** 2).mean()), sin_max=float((1 - sv.min() ** 2).clamp(min=0).sqrt()), cos_min=float(sv.min())))
    return dict(probe=pchg, drift=drift, final_avg=float(accm[-1].mean()), forgetting=forget, last_task=float(accm[-1, -1]), first_task_final=float(accm[-1, 0]), bits=bits, p=p, ranks=ranks, lranks=lranks, acc=accm.tolist(), pol=POL.tolist())


if __name__ == "__main__":
    t0 = time.time(); r = run(); r["args"] = vars(a); r["time"] = time.time() - t0
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("acc", "args", "pol")}))
    if a.out: os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(r, open(a.out + ".json", "w"))
