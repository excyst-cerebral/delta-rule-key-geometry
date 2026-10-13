"""Stage 0 (dedupe) and Stage 1 (automatic keyword screen) exactly as in review/protocol.md.  stdlib only."""
import json, glob, re, os, collections
HERE = os.path.dirname(os.path.abspath(__file__))
norm = lambda t: re.sub(r"[^a-z0-9]", "", (t or "").lower())
recs, per_src = {}, collections.Counter(); raw_total = 0; hits = collections.defaultdict(set)
for f in sorted(glob.glob(HERE + "/raw/*_S*.json")):
    src, qid = os.path.basename(f)[:-5].split("_"); L = json.load(open(f)); per_src[(qid, src)] = len(L); raw_total += len(L)
    for r in L:
        keys = [("ax:" + r["arxiv"]) if r.get("arxiv") else None, ("doi:" + r["doi"].lower().split("doi.org/")[-1]) if r.get("doi") else None, "t:" + norm(r["title"])]
        keys = [k for k in keys if k and k != "t:"]
        k0 = next((k for k in keys if k in recs), None)
        if k0 is None: k0 = keys[0]; recs[k0] = dict(r, queries=set(), srcs=set(), alias=set())
        rr = recs[k0]; rr["queries"].add(qid); rr["srcs"].add(src); rr["alias"].update(keys)
        if not rr["abstract"] and r["abstract"]: rr["abstract"] = r["abstract"]
        if not rr.get("arxiv") and r.get("arxiv"): rr["arxiv"] = r["arxiv"]
        if (rr.get("cited") is None) and r.get("cited") is not None: rr["cited"] = r["cited"]
        if r["src"] == "openalex" and r.get("venue") and rr.get("venue") in (None, "", "arXiv"): rr["venue"] = r["venue"]
        for k in keys: hits[k].add(k0)
# merge records sharing any alias (one pass)
parent = {}
def find(x):
    while parent.get(x, x) != x: x = parent[x]
    return x
for k, ids in hits.items():
    ids = list(ids)
    for a in ids[1:]: parent[find(a)] = find(ids[0])
merged = {}
for k0, r in recs.items():
    root = find(k0)
    if root not in merged: merged[root] = r
    else:
        m = merged[root]; m["queries"] |= r["queries"]; m["srcs"] |= r["srcs"]
        if not m["abstract"] and r["abstract"]: m["abstract"] = r["abstract"]
        if not m.get("arxiv") and r.get("arxiv"): m["arxiv"] = r["arxiv"]
        if m.get("cited") is None and r.get("cited") is not None: m["cited"] = r["cited"]
        if r.get("venue") and m.get("venue") in (None, "", "arXiv"): m["venue"] = r["venue"]
U = list(merged.values())
T = lambda r: (r["title"] + " . " + (r["abstract"] or "")).lower()
R = {
 "R1": lambda t: re.search(r"mamba|state[- ]space|\bssms?\b|linear attention|fast[- ]weight|delta rule|deltanet|rwkv|test-time training|linear recurrent|linear transformer", t) and re.search(r"continual|lifelong|life-long|forgetting|incremental", t),
 "R2": lambda t: re.search(r"continual|catastroph|forgetting|sequential learning|lifelong", t) and re.search(r"linear regression|linear model|kaczmarz|alternating projection|overparameter|neural tangent|\bntk\b|lazy|kernel regime|least[- ]squares", t),
 "R3": lambda t: re.search(r"continual|catastroph|forgetting|lifelong", t) and re.search(r"projection|null[- ]space|orthogonal|subspace|stability[- ]plasticity|gradient", t) and re.search(r"theor|bound|guarantee|provab|prove|analy[sz]|trade-?off|memory-efficient|compress|quantiz|storage|low-rank|adapter|lora", t),
 "R4": lambda t: re.search(r"continual|lifelong|catastroph|sequential learning", t) and re.search(r"memory|storage|\bbits?\b", t) and re.search(r"lower bound|communication complexity|information[- ]theoretic|np-hard|impossib|perfect memory|hardness|\bpac\b", t),
 "R5": lambda t: re.search(r"delta rule|associative memory|fast[- ]weight|linear attention|linear transformer", t) and re.search(r"capacity|interference|orthogon|crosstalk|retriev|recall", t),
 "R5b": lambda t: re.search(r"online least squares|online regression|test-time regression|online learning|least[- ]squares", t) and re.search(r"sequence model|state[- ]space|linear attention|recurrent|associative memory", t),
 "R6": lambda t: re.search(r"lora|low-rank|adapter|parameter isolation|task-specific|expert|prompt", t) and re.search(r"continual|lifelong|incremental", t) and re.search(r"orthogonal|interference|forgetting|subspace|guarantee|theor", t),
 "R7": lambda t: re.search(r"grassmann|stiefel", t) and re.search(r"covering|metric entropy|packing|volume", t),
 "R8": lambda t: re.search(r"plasticity", t) and re.search(r"continual|deep learning|lifelong", t),
}
TIER_A = {"R1", "R2", "R4", "R5", "R5b", "R7"}
for r in U:
    t = T(r); r["rules"] = sorted(k for k, f in R.items() if f(t)); r["queries"] = sorted(r["queries"]); r["srcs"] = sorted(r["srcs"]); r.pop("alias", None)
    r["tier"] = "A" if set(r["rules"]) & TIER_A else ("B" if r["rules"] else None)
kept = [r for r in U if r["rules"]]
json.dump(U, open(HERE + "/stage0_unique.json", "w")); json.dump(kept, open(HERE + "/stage1_kept.json", "w"))
stats = dict(raw_records=raw_total, per_query_source={f"{q}_{s}": n for (q, s), n in sorted(per_src.items())}, unique_after_dedupe=len(U), no_abstract=sum(1 for r in U if not r["abstract"]),
             stage1_kept=len(kept), tierA=sum(r["tier"] == "A" for r in kept), tierB=sum(r["tier"] == "B" for r in kept), auto_excluded=len(U) - len(kept),
             kept_by_rule={k: sum(k in r["rules"] for r in kept) for k in R}, by_source=dict(collections.Counter("+".join(r["srcs"]) for r in U)))
json.dump(stats, open(HERE + "/stage_counts.json", "w"), indent=1); print(json.dumps(stats, indent=1))
