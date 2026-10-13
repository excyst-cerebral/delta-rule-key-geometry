import json
import numpy as np

R = json.load(open("results_gdn.json"))
per = R["per_head"]
print("perplexity (sanity):", {k: round(v, 2) for k, v in R["ppl"].items()})
NL, H = 24, 4


def get(dom, layers, key):
    return np.array([per[f"{dom}|{l}|{h}"][key] for l in layers for h in range(H)])


def med(dom, layers, key):
    return float(np.median(get(dom, layers, key)))


for dom in ("prose", "code"):
    print(f"\n================ {dom} ================")
    print("per-layer medians over 4 heads")
    print(f"{'L':>2} {'halflife':>9} {'beta':>5} {'surpr':>6} | {'c(1)':>5} {'c(16)':>5} {'c(64)':>5} {'c(256)':>6} | "
          f"{'mu64':>5} {'effrk64':>7} {'lmin64':>8} | {'A@64':>6} {'ret@64':>6} {'D@64':>6} {'B@64':>7}")
    for l in range(NL):
        L = [l]
        print(f"{l:2d} {med(dom,L,'halflife_med'):9.1f} {med(dom,L,'beta_mean'):5.2f} {med(dom,L,'surprise_rel'):6.2f} | "
              f"{med(dom,L,'c1'):5.2f} {med(dom,L,'c16'):5.2f} {med(dom,L,'c64'):5.2f} {med(dom,L,'c256'):6.2f} | "
              f"{med(dom,L,'w64_mu'):5.2f} {med(dom,L,'w64_eff_rank'):7.1f} {med(dom,L,'w64_lam_min'):8.1e} | "
              f"{med(dom,L,'p64_A'):6.3f} {med(dom,L,'p64_ret'):6.3f} {med(dom,L,'p64_D'):6.3f} {med(dom,L,'p64_B'):7.3f}")
    allL = range(NL)
    print("\nALL layers/heads (median [p10, p90] over the 96 heads)")
    for key, label in [("halflife_med", "decay half-life (tokens)"), ("beta_mean", "beta (write strength)"),
                       ("surprise_rel", "surprise ||e||/||v||"), ("c1", "|<k_t,k_{t-1}>|"), ("c16", "|<k_t,k_{t-16}>|"),
                       ("c64", "|<k_t,k_{t-64}>|"), ("c256", "|<k_t,k_{t-256}>|"),
                       ("w16_mu", "coherence mu (n=16)"), ("w64_mu", "coherence mu (n=64)"),
                       ("w16_eff_rank", "key eff. rank (n=16)"), ("w64_eff_rank", "key eff. rank (n=64)"),
                       ("w64_lam_min", "lambda_min(G) (n=64)"), ("w16_lam_min", "lambda_min(G) (n=16)"),
                       ("w16_sv0", "||V||_op, unit rows (n=16)"), ("w64_sv0", "||V||_op, unit rows (n=64)"),
                       ("w16_F_ideal", "ideal a=b=1 worst err (n=16)"), ("w16_thm5", "Thm5 bound (n=16)"),
                       ("w64_F_ideal", "ideal a=b=1 worst err (n=64)"), ("w64_thm5", "Thm5 bound (n=64)")]:
        x = get(dom, allL, key)
        print(f"  {label:34s} {np.median(x):10.3g}  [{np.percentile(x,10):9.3g}, {np.percentile(x,90):9.3g}]")
    print("\nDecay vs interference at horizon D (median [p10,p90] over heads); retention is ||r_{s+D}||/||r_s||")
    print(f"  {'D':>4} {'pure decay A':>22} {'actual retention':>22} {'interference D/||r||':>24} {'Thm1 bound/||r||':>22} {'bound holds':>12} {'lemma resid max':>16}")
    for dl in (16, 64, 256):
        f = lambda k: get(dom, allL, f"p{dl}_{k}")
        cell = lambda x: f"{np.median(x):.3f} [{np.percentile(x,10):.3f},{np.percentile(x,90):.3f}]"
        print(f"  {dl:4d} {cell(f('A')):>22} {cell(f('ret')):>22} {cell(f('D')):>24} {cell(f('B')):>22} "
              f"{f('bound_ok').min():12.2f} {f('lemma_max').max():16.1e}")
