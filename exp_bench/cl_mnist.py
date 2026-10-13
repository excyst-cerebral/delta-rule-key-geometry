"""Standard continual-learning benchmark for a small selective-free (real diagonal) SSM: Split-MNIST and Permuted-MNIST.  Torch + numpy only.
Each 28x28 image is a sequence of 28 steps of 28 values.  Model: Linear(28->C) -> SSM -> SiLU -> Linear(C->C) -> SSM -> last step -> Linear(C->10).
Training always uses the full 10-way cross-entropy (no task id, no masking).  Evaluation:
  split: class-incremental (argmax over all 10 logits) and task-incremental (argmax over the task's 2 classes, task id given at test time only)
  perm : domain-incremental (10-way, no task id)
Learners (plain SGD + momentum, reset per task): joint (upper bound: all tasks trained together, same total compute) | ft | ewc | replay | adapter (task-specific head + rank-r LoRA on l1 and mix, frozen backbone after task 0, TASK ID GIVEN at test time) | gpm (layerwise activation-subspace projection; --ssm free|freeze|fixed; fixed = recurrence never trained, applies to any method) | ogd (gradient subspace)
Memory is counted in bits: replay stores 8-bit pixels + 4-bit label; EWC stores 2 float32 copies of the parameters per task; projections store q-bit basis entries.
"""
import argparse, gzip, json, math, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

p_ = argparse.ArgumentParser()
p_.add_argument("--bench", default="split"); p_.add_argument("--method", default="ft"); p_.add_argument("--seed", type=int, default=0)
p_.add_argument("--rho", type=float, default=0.99); p_.add_argument("--q", type=int, default=32); p_.add_argument("--ssm", default="free")
p_.add_argument("--ewc", type=float, default=100.0); p_.add_argument("--k", type=int, default=50); p_.add_argument("--C", type=int, default=32); p_.add_argument("--N", type=int, default=8)
p_.add_argument("--epochs", type=int, default=2); p_.add_argument("--lr", type=float, default=0.01); p_.add_argument("--mom", type=float, default=0.9)
p_.add_argument("--rank", type=int, default=4); p_.add_argument("--ntrain", type=int, default=0); p_.add_argument("--ntest", type=int, default=0); p_.add_argument("--ns", type=int, default=300)
p_.add_argument("--data", default="data_mnist"); p_.add_argument("--threads", type=int, default=1); p_.add_argument("--out", default="")
a = p_.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)


def idx(path):
    b = gzip.open(path, "rb").read(); nd = b[3]; shape = tuple(int.from_bytes(b[4 + 4 * i:8 + 4 * i], "big") for i in range(nd))
    return np.frombuffer(b, dtype=np.uint8, offset=4 + 4 * nd).reshape(shape)


D = a.data
Xtr = idx(f"{D}/train-images-idx3-ubyte.gz").astype(np.float32) / 255.; Ytr = idx(f"{D}/train-labels-idx1-ubyte.gz").astype(np.int64)
Xte = idx(f"{D}/t10k-images-idx3-ubyte.gz").astype(np.float32) / 255.; Yte = idx(f"{D}/t10k-labels-idx1-ubyte.gz").astype(np.int64)
assert Xtr.shape == (60000, 28, 28) and Xte.shape == (10000, 28, 28) and Ytr.max() == 9


def make_tasks():
    tasks = []
    if a.bench == "split":
        for t in range(5):
            cl = (2 * t, 2 * t + 1); mtr = np.isin(Ytr, cl); mte = np.isin(Yte, cl)
            tasks.append(dict(xtr=Xtr[mtr], ytr=Ytr[mtr], xte=Xte[mte], yte=Yte[mte], classes=cl))
    else:
        for t in range(10):
            perm = np.arange(784) if t == 0 else np.random.default_rng(1234 + 17 * a.seed + t).permutation(784)
            sub = np.random.default_rng(99 + a.seed).permutation(60000)[:(a.ntrain or 6000)]; te = slice(0, a.ntest or 3000)
            tasks.append(dict(xtr=Xtr[sub].reshape(-1, 784)[:, perm].reshape(-1, 28, 28), ytr=Ytr[sub], xte=Xte[te].reshape(-1, 784)[:, perm].reshape(-1, 28, 28), yte=Yte[te], classes=tuple(range(10))))
    for tk in tasks:
        if a.ntrain and a.bench == "split": tk["xtr"], tk["ytr"] = tk["xtr"][:a.ntrain], tk["ytr"][:a.ntrain]
        for k_ in ("xtr", "xte"): tk[k_] = torch.from_numpy(np.ascontiguousarray(tk[k_]))
        for k_ in ("ytr", "yte"): tk[k_] = torch.from_numpy(np.ascontiguousarray(tk[k_]))
    return tasks


class SSM(nn.Module):
    def __init__(s, C, N):
        super().__init__(); lam = torch.linspace(0.3, 0.98, C * N)[torch.randperm(C * N)].view(C, N)
        s.raw = nn.Parameter(torch.logit(lam)); s.B = nn.Parameter(torch.randn(C, N) / math.sqrt(N)); s.Cm = nn.Parameter(torch.randn(C, N) / math.sqrt(N)); s.D = nn.Parameter(torch.zeros(C))
    def forward(s, u):
        lam = torch.sigmoid(s.raw); h = u.new_zeros(u.shape[0], *s.raw.shape); ys = []
        for t in range(u.shape[1]):
            h = lam * h + s.B * u[:, t, :, None]; ys.append((h * s.Cm).sum(-1) + s.D * u[:, t])
        return torch.stack(ys, 1)


class Net(nn.Module):
    def __init__(s, C, N):
        super().__init__(); s.l1 = nn.Linear(28, C); s.s1 = SSM(C, N); s.mix = nn.Linear(C, C); s.s2 = SSM(C, N); s.head = nn.Linear(C, 10)
    def forward(s, x):
        h = F.silu(s.s1(s.l1(x))); h = s.s2(s.mix(h))[:, -1]; return s.head(h)


LIN = ("l1", "mix", "head")
flat = lambda m: torch.cat([p.reshape(-1) for p in m.parameters()])
flat_grad = lambda m: torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in m.parameters()])


def set_grad(m, g):
    i = 0
    for p in m.parameters(): n = p.numel(); p.grad = g[i:i + n].view_as(p).clone(); i += n


@torch.no_grad()
def evaluate(model, tk):
    out = torch.cat([model(tk["xte"][i:i + 1000]) for i in range(0, len(tk["xte"]), 1000)]); y = tk["yte"]
    full = (out.argmax(-1) == y).float().mean().item(); m = torch.full_like(out, -1e9); m[:, list(tk["classes"])] = out[:, list(tk["classes"])]
    return full, (m.argmax(-1) == y).float().mean().item()


def per_sample_grads(model, x, y):
    G = []
    for i in range(len(x)):
        model.zero_grad(); F.cross_entropy(model(x[i:i + 1]), y[i:i + 1]).backward(); G.append(flat_grad(model).clone())
    model.zero_grad(); return torch.stack(G)


@torch.no_grad()
def collect_acts(model, x):
    store = {}; hs = [getattr(model, nm).register_forward_pre_hook(lambda m, inp, nm=nm: store.__setitem__(nm, inp[0].reshape(-1, inp[0].shape[-1]).clone())) for nm in LIN]
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
    tasks = make_tasks(); M = len(tasks); model = Net(a.C, a.N); p = sum(x.numel() for x in model.parameters())
    ssm_idx = torch.cat([torch.full((q_.numel(),), float(any(q_ is r_ for sm in (model.s1, model.s2) for r_ in sm.parameters()))) for q_ in model.parameters()]).bool()
    lb = {nm: torch.zeros(getattr(model, nm).weight.shape[1] + 1, 0) for nm in LIN}; basis = torch.zeros(p, 0); anchors = []; buf = []; bits = 0; lranks = []; oranks = []
    acc_full = torch.zeros(M, M); acc_task = torch.zeros(M, M); g = torch.Generator().manual_seed(1000 + a.seed)
    comb = dict(xtr=torch.cat([tk["xtr"] for tk in tasks]), ytr=torch.cat([tk["ytr"] for tk in tasks])) if a.method == "joint" else None
    for t in range(M):
        if a.method == "joint" and t < M - 1: continue
        tk = comb if a.method == "joint" else tasks[t]; n = len(tk["xtr"]); vel = torch.zeros(p)
        for ep in range(a.epochs):
            order = torch.randperm(n, generator=g)
            for i in range(0, n - 63, 128):
                b = order[i:i + 128]; x, y = tk["xtr"][b], tk["ytr"][b]; model.zero_grad(); loss = F.cross_entropy(model(x), y)
                if a.method == "replay" and buf:
                    bx = torch.cat([u[0] for u in buf]); by = torch.cat([u[1] for u in buf]); j = torch.randint(0, len(bx), (128,), generator=g); loss = loss + F.cross_entropy(model(bx[j]), by[j])
                if a.method == "ewc" and anchors:
                    th = flat(model)
                    for (ts, Fi) in anchors: loss = loss + a.ewc * (Fi * (th - ts) ** 2).sum()
                loss.backward(); gr = flat_grad(model); gr = gr * min(1.0, 5.0 / (float(gr.norm()) + 1e-12)); set_grad(model, gr)
                if a.method == "gpm":
                    project_linear(model, lb); gr = flat_grad(model)
                    set_grad(model, gr)
                if a.ssm == "fixed" or (a.method == "gpm" and a.ssm == "freeze" and t > 0):
                    gr = flat_grad(model); gr[ssm_idx] = 0; set_grad(model, gr)
                gr = flat_grad(model)
                if a.method == "ogd" and basis.shape[1]: gr = gr - basis @ (basis.T @ gr)
                vel = a.mom * vel + gr
                with torch.no_grad():
                    i0 = 0
                    for prm in model.parameters(): m_ = prm.numel(); prm -= a.lr * vel[i0:i0 + m_].view_as(prm); i0 += m_
        sel = torch.randperm(n, generator=g)[:a.ns]; xs, ys = tk["xtr"][sel], tk["ytr"][sel]
        if a.method == "gpm":
            acts = collect_acts(model, xs); rr = []
            for nm in LIN:
                A = acts[nm]; B = lb[nm]; tot = float((A ** 2).sum()); old = float(((A @ B) ** 2).sum()) if B.shape[1] else 0.0
                R = A - A @ B @ B.T if B.shape[1] else A
                U, S, Vh = torch.linalg.svd(R, full_matrices=False); cum = (old + torch.cumsum(S ** 2, 0)) / tot; r = int((cum < a.rho).sum()) + 1
                new = gram_schmidt(B, quant(Vh[:r].T, a.q)); lb[nm] = torch.cat([B, new], 1); bits += new.shape[1] * lb[nm].shape[0] * a.q; rr.append(new.shape[1])
            lranks.append(rr)
        if a.method == "ogd":
            G = per_sample_grads(model, xs[:200], ys[:200]); R = G - G @ basis @ basis.T if basis.shape[1] else G
            U, S, Vh = torch.linalg.svd(R, full_matrices=False); e = torch.cumsum(S ** 2, 0) / (S ** 2).sum(); r = int((e < a.rho).sum()) + 1
            new = gram_schmidt(basis, quant(Vh[:r].T, a.q)); basis = torch.cat([basis, new], 1); bits += new.shape[1] * p * a.q; oranks.append(new.shape[1])
        if a.method == "ewc":
            G = per_sample_grads(model, xs[:200], ys[:200]); anchors.append((flat(model).detach().clone(), (G ** 2).mean(0))); bits += 2 * p * 32
        if a.method == "replay":
            buf.append((tk["xtr"][sel[:a.k]], tk["ytr"][sel[:a.k]])); bits += a.k * (784 * 8 + 4)
        for j in range(M): acc_full[t, j], acc_task[t, j] = evaluate(model, tasks[j])
    res = {}
    for name, A in (("full", acc_full), ("task", acc_task)):
        res[name] = dict(final_avg=float(A[-1].mean()), forgetting=0.0 if a.method == "joint" else float(sum(A[:M - 1, j].max() - A[-1, j] for j in range(M - 1)) / (M - 1)), last_task=float(A[-1, -1]), acc=A.tolist())
    return dict(res=res, bits=bits, p=p, lranks=lranks, oranks=oranks, tasks=M)


def lora(x, d): return (x @ d["V"].T) @ d["U"].T


def fwd(model, x, ad):
    h = model.l1(x)
    if ad is not None and "l1" in ad: h = h + lora(x, ad["l1"])
    h = F.silu(model.s1(h)); z = model.mix(h)
    if ad is not None and "mix" in ad: z = z + lora(h, ad["mix"])
    h = model.s2(z)[:, -1]
    return ad["head"](h) if ad is not None else model.head(h)


@torch.no_grad()
def evaluate_ad(model, tk, ad):
    out = torch.cat([fwd(model, tk["xte"][i:i + 1000], ad) for i in range(0, len(tk["xte"]), 1000)]); y = tk["yte"]
    full = (out.argmax(-1) == y).float().mean().item(); m = torch.full_like(out, -1e9); m[:, list(tk["classes"])] = out[:, list(tk["classes"])]
    return full, (m.argmax(-1) == y).float().mean().item()


def new_adapter(C):
    ad = dict(head=nn.Linear(C, 10))
    if a.rank > 0:
        ad["l1"] = dict(V=nn.Parameter(torch.randn(a.rank, 28) * 0.1), U=nn.Parameter(torch.zeros(C, a.rank)))
        ad["mix"] = dict(V=nn.Parameter(torch.randn(a.rank, C) / math.sqrt(C)), U=nn.Parameter(torch.zeros(C, a.rank)))
    return ad


def ad_params(ad):
    ps = list(ad["head"].parameters())
    for k in ("l1", "mix"):
        if k in ad: ps += [ad[k]["V"], ad[k]["U"]]
    return ps


def run_adapter():
    tasks = make_tasks(); M = len(tasks); model = Net(a.C, a.N); ads = [None] * M; bits = 0
    acc_full = torch.zeros(M, M); acc_task = torch.zeros(M, M); g = torch.Generator().manual_seed(1000 + a.seed); ref = None; maxchg = 0.0
    for t in range(M):
        tk = tasks[t]; n = len(tk["xtr"])
        if t == 0: params = list(model.parameters())
        else:
            for q_ in model.parameters(): q_.requires_grad_(False)
            ads[t] = new_adapter(a.C); params = ad_params(ads[t]); bits += sum(q_.numel() for q_ in params) * 32
        vel = [torch.zeros_like(q_) for q_ in params]
        for ep in range(a.epochs):
            order = torch.randperm(n, generator=g)
            for i in range(0, n - 63, 128):
                b = order[i:i + 128]; x, y = tk["xtr"][b], tk["ytr"][b]
                for q_ in params: q_.grad = None
                loss = F.cross_entropy(fwd(model, x, ads[t]), y); loss.backward()
                gn = torch.sqrt(sum((q_.grad ** 2).sum() for q_ in params)); sc = min(1.0, 5.0 / (float(gn) + 1e-12))
                with torch.no_grad():
                    for q_, v_ in zip(params, vel): v_.mul_(a.mom).add_(q_.grad * sc); q_.sub_(a.lr * v_)
        if t >= 1:
            with torch.no_grad():
                out = fwd(model, tasks[1]["xte"][:500], ads[1])
                if t == 1: ref = out.clone()
                else: maxchg = max(maxchg, float((out - ref).abs().max()))
        for j in range(t + 1): acc_full[t, j], acc_task[t, j] = evaluate_ad(model, tasks[j], ads[j])
        for j in range(t + 1, M): acc_full[t, j], acc_task[t, j] = evaluate_ad(model, tasks[j], ads[j] if ads[j] is not None else new_adapter(a.C)) if False else (0.0, 0.0)
    res = {}
    for name, A in (("full", acc_full), ("task", acc_task)):
        res[name] = dict(final_avg=float(A[-1].mean()), forgetting=float(sum(A[:M - 1, j].max() - A[-1, j] for j in range(M - 1)) / (M - 1)), last_task=float(A[-1, -1]), acc=A.tolist())
    return dict(res=res, bits=bits, p=sum(q_.numel() for q_ in model.parameters()), lranks=[], oranks=[], tasks=M, max_old_logit_change=maxchg, adapter_params=sum(q_.numel() for q_ in ad_params(ads[1])) if M > 1 else 0)


if __name__ == "__main__":
    t0 = time.time(); r = run_adapter() if a.method == "adapter" else run(); r["args"] = vars(a); r["time"] = time.time() - t0
    print(json.dumps(dict(bench=a.bench, method=a.method, seed=a.seed, full=round(r["res"]["full"]["final_avg"], 3), fgt=round(r["res"]["full"]["forgetting"], 3), task=round(r["res"]["task"]["final_avg"], 3), bits=r["bits"], time=round(r["time"]))))
    if a.out: os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(r, open(a.out + ".json", "w"))
