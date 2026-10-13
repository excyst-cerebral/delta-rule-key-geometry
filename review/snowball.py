"""Backward/forward citation chasing from the pre-specified seeds (protocol.md).  stdlib only."""
import json, urllib.request, urllib.parse, time, re, os
HERE = os.path.dirname(os.path.abspath(__file__))
def get(url):
    for k in range(4):
        try: return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "systematic-review-script"}), timeout=60).read())
        except Exception as e: time.sleep(4 * (k + 1)); err = e
    raise err
SEEDS = {  # title keyword for OpenAlex lookup -> label
 "How catastrophic can catastrophic forgetting be in linear regression": "Evron2022", "From Continual Learning to SGD and Back": "Evron2026", "Theory on Forgetting and Generalization of Continual Learning": "Lin2023",
 "Memory Bounds for Continual Learning": "Chen2022", "Optimal Continual Learning has Perfect Memory and is NP-hard": "Knoblauch2020", "Gradient Projection Memory for Continual Learning": "Saha2021",
 "Training Networks in Null Space of Feature Covariance for Continual Learning": "Wang2021", "Mamba-CL: Optimizing Selective State Space Model in Null Space": "Cheng2024", "Exemplar-Free Continual Learning for State Space Models": "Lee2025",
 "Longhorn: State Space Models are Amortized Online Learners": "Liu2025", "Test-time regression: a unifying framework for designing sequence models with associative memory": "Wang2025", "Linear Transformers Are Secretly Fast Weight Programmers": "Schlag2021",
 "PMF-CL: Pareto-Minimal-Forgetting Continual Learner": "Nair2026"}
out = {}; log = {}
norm = lambda t: re.sub(r"[^a-z0-9]", "", (t or "").lower())
for title, lab in SEEDS.items():
    j = get("https://api.openalex.org/works?filter=title.search:" + urllib.parse.quote(f'"{title[:80]}"') + "&per-page=15&select=id,title,publication_year,referenced_works,cited_by_count"); time.sleep(0.3)
    ms = [w for w in j["results"] if norm(w["title"]).startswith(norm(title)[:40])]   # OpenAlex stores arXiv / proceedings versions as separate records: use all of them
    if not ms: log[lab] = "not found"; continue
    ids = [w["id"].split("/")[-1] for w in ms]; refs = sorted({r.split("/")[-1] for w in ms for r in w["referenced_works"]})
    log[lab] = dict(openalex=ids, title=ms[0]["title"], cited_by=sum(w["cited_by_count"] for w in ms), n_records=len(ms), n_refs=len(refs))
    out[lab] = dict(ids=ids, refs=refs)
# forward: works citing a seed (all pages, up to 1500 each)
fwd = {}
for lab, d in out.items():
    cur = "*"; n = 0
    while cur and n < 1500:
        j = get(f"https://api.openalex.org/works?filter=cites:{'|'.join(d['ids'])}&per-page=200&cursor={cur}&select=id,doi,title,publication_year,abstract_inverted_index,cited_by_count,primary_location,ids"); time.sleep(0.3)
        for w in j["results"]:
            inv = w.get("abstract_inverted_index") or {}; pos = {}
            for t, ps in inv.items():
                for p in ps: pos[p] = t
            fwd.setdefault(w["id"].split("/")[-1], dict(title=w.get("title") or "", year=w.get("publication_year"), doi=w.get("doi"), abstract=" ".join(pos[i] for i in sorted(pos)), cited=w.get("cited_by_count"), venue=((w.get("primary_location") or {}).get("source") or {}).get("display_name") or "", cites=[]))["cites"].append(lab)
        n += len(j["results"]); cur = j["meta"].get("next_cursor")
        if not j["results"]: break
    log[lab]["forward_fetched"] = n
# backward: referenced works (fetch metadata in batches of 50)
bwd = {}
for lab, d in out.items():
    ids = d["refs"]
    for i in range(0, len(ids), 50):
        j = get("https://api.openalex.org/works?filter=openalex:" + "|".join(ids[i:i + 50]) + "&per-page=50&select=id,doi,title,publication_year,abstract_inverted_index,cited_by_count,primary_location"); time.sleep(0.3)
        for w in j["results"]:
            inv = w.get("abstract_inverted_index") or {}; pos = {}
            for t, ps in inv.items():
                for p in ps: pos[p] = t
            bwd.setdefault(w["id"].split("/")[-1], dict(title=w.get("title") or "", year=w.get("publication_year"), doi=w.get("doi"), abstract=" ".join(pos[i] for i in sorted(pos)), cited=w.get("cited_by_count"), venue=((w.get("primary_location") or {}).get("source") or {}).get("display_name") or "", cited_by_seed=[]))["cited_by_seed"].append(lab)
json.dump(dict(log=log, forward=fwd, backward=bwd), open(HERE + "/snowball_raw.json", "w"))
print(json.dumps(log, indent=1)); print("forward unique", len(fwd), "backward unique", len(bwd))
