"""Stage A (torch only): final-state features of a random-init SSM on each task's data, saved for the numpy readout-law test."""
import sys, math, numpy as np, torch
sys.argv = [sys.argv[0]] + ["--arch", sys.argv[1], "--seed", sys.argv[2]]
import importlib.util
spec = importlib.util.spec_from_file_location("cl", "exp_cl/cl_ssm.py"); cl = importlib.util.module_from_spec(spec); spec.loader.exec_module(cl)
a = cl.a; torch.manual_seed(a.seed)
model = cl.SSM(a.arch, a.C, a.N, a.de); g = torch.Generator().manual_seed(555 + a.seed); out = {}
with torch.no_grad():
    for t in range(a.tasks):
        x, y = cl.batch(t, 2000, g); out[f"phi{t}"] = model.features(cl.E[x]).numpy(); out[f"y{t}"] = (2 * y.float() - 1).numpy()
np.savez(f"exp_cl/runs/feat_{a.arch}_s{a.seed}.npz", **out)
