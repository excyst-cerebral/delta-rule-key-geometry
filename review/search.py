"""Run the pre-specified searches (review/protocol.md) against arXiv and OpenAlex.  stdlib only.  Logs total hits and truncation per query."""
import json, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET, os, sys, re
HERE = os.path.dirname(os.path.abspath(__file__)); RAW = HERE + "/raw"; os.makedirs(RAW, exist_ok=True)
UA = {"User-Agent": "systematic-review-script (mailto:none)"}
Q = lambda *terms: " OR ".join(terms)
CL = Q('"continual learning"', '"catastrophic forgetting"', 'lifelong', '"class-incremental"', '"incremental learning"')
QUERIES = {
 "S1": dict(ax=f'(ti:mamba OR ti:"state space" OR ti:"state-space" OR ti:SSM) AND ({Q(*[f"abs:{t}" for t in ["\"continual learning\"", "\"catastrophic forgetting\"", "lifelong", "\"class-incremental\"", "\"incremental learning\""]])})',
            oa='title.search:(mamba OR "state space" OR "state-space" OR SSM),title_and_abstract.search:("continual learning" OR "catastrophic forgetting" OR lifelong OR "class-incremental" OR "incremental learning")'),
 "S2": dict(ax='(abs:"linear attention" OR abs:"fast weight" OR abs:"fast weights" OR abs:"delta rule" OR abs:DeltaNet OR abs:RWKV OR abs:"test-time training" OR abs:"linear transformer") AND (abs:"catastrophic forgetting" OR abs:"continual learning" OR abs:lifelong)',
            oa='title_and_abstract.search:("linear attention" OR "fast weight" OR "fast weights" OR "delta rule" OR DeltaNet OR RWKV OR "test-time training" OR "linear transformer"),title_and_abstract.search:("catastrophic forgetting" OR "continual learning" OR lifelong)'),
 "S3": dict(ax='(abs:"continual learning" OR abs:"catastrophic forgetting" OR abs:"sequential learning") AND (abs:"linear regression" OR abs:"linear models" OR abs:"linear model") AND (abs:Kaczmarz OR abs:"alternating projection" OR abs:theory OR abs:"upper bound" OR abs:"lower bound" OR abs:overparameterized OR abs:rates)',
            oa='title_and_abstract.search:("continual learning" OR "catastrophic forgetting" OR "sequential learning"),title_and_abstract.search:("linear regression" OR "linear models" OR "linear model"),title_and_abstract.search:(Kaczmarz OR "alternating projection" OR theory OR "upper bound" OR "lower bound" OR overparameterized OR rates)'),
 "S4a": dict(ax='abs:continual AND abs:Kaczmarz', oa='title_and_abstract.search:(continual AND Kaczmarz)'),
 "S4b": dict(ax='abs:forgetting AND abs:"neural tangent kernel" AND abs:continual', oa='title_and_abstract.search:(forgetting AND "neural tangent kernel" AND continual)'),
 "S5": dict(ax='(abs:"gradient projection" OR abs:"orthogonal gradient" OR abs:"null space" OR abs:"null-space" OR abs:"orthogonal projection" OR abs:subspace) AND (abs:"continual learning" OR abs:"catastrophic forgetting")',
            oa='title_and_abstract.search:("gradient projection" OR "orthogonal gradient" OR "null space" OR "null-space" OR "orthogonal projection" OR subspace),title_and_abstract.search:("continual learning" OR "catastrophic forgetting")'),
 "S6": dict(ax='(abs:"stability-plasticity" OR abs:"stability plasticity") AND (abs:theory OR abs:theoretical OR abs:bound OR abs:"trade-off" OR abs:tradeoff OR abs:Pareto)',
            oa='title_and_abstract.search:("stability-plasticity" OR "stability plasticity"),title_and_abstract.search:(theory OR theoretical OR bound OR "trade-off" OR tradeoff OR Pareto)'),
 "S7": dict(ax='abs:"continual learning" AND (abs:memory OR abs:"memory requirement") AND (abs:"lower bound" OR abs:"communication complexity" OR abs:"information-theoretic" OR abs:"NP-hard" OR abs:"perfect memory" OR abs:impossibility)',
            oa='title_and_abstract.search:"continual learning",title_and_abstract.search:(memory),title_and_abstract.search:("lower bound" OR "communication complexity" OR "information-theoretic" OR "NP-hard" OR "perfect memory" OR impossibility)'),
 "S8": dict(ax='(abs:"delta rule" OR abs:"associative memory" OR abs:"fast weight") AND (abs:"linear attention" OR abs:"linear transformer" OR abs:recurrent OR abs:"state space") AND (abs:capacity OR abs:interference OR abs:orthogonal OR abs:key)',
            oa='title_and_abstract.search:("delta rule" OR "associative memory" OR "fast weight"),title_and_abstract.search:("linear attention" OR "linear transformer" OR recurrent OR "state space"),title_and_abstract.search:(capacity OR interference OR orthogonal)'),
 "S9": dict(ax='(abs:"online least squares" OR abs:"online regression" OR abs:"test-time regression") AND (abs:"sequence model" OR abs:"state space" OR abs:"linear attention" OR abs:recurrent)',
            oa='title_and_abstract.search:("online least squares" OR "online regression" OR "test-time regression"),title_and_abstract.search:("sequence model" OR "state space" OR "linear attention" OR recurrent)'),
 "S10": dict(ax='(abs:LoRA OR abs:"low-rank adaptation" OR abs:adapter OR abs:adapters OR abs:"parameter isolation" OR abs:"task-specific") AND abs:"continual learning" AND (abs:orthogonal OR abs:forgetting OR abs:interference OR abs:"zero forgetting")',
             oa='title_and_abstract.search:(LoRA OR "low-rank adaptation" OR adapter OR adapters OR "parameter isolation" OR "task-specific"),title_and_abstract.search:"continual learning",title_and_abstract.search:(orthogonal OR interference OR "zero forgetting")'),
 "S11": dict(ax='(abs:Grassmann OR abs:Grassmannian OR ti:Grassmann OR ti:Grassmannian) AND (abs:"covering number" OR abs:"covering numbers" OR abs:"metric entropy" OR abs:packing)',
             oa='title_and_abstract.search:(Grassmann OR Grassmannian),title_and_abstract.search:("covering number" OR "covering numbers" OR "metric entropy" OR packing)'),
 "S12": dict(ax='(abs:"loss of plasticity" OR abs:"plasticity loss") AND (abs:continual OR abs:"deep learning")', oa='title_and_abstract.search:("loss of plasticity" OR "plasticity loss"),title_and_abstract.search:(continual OR "deep learning")'),
 "S13": dict(ax='(abs:"exemplar-free" OR abs:"rehearsal-free" OR abs:"replay-free" OR abs:"memory-free") AND abs:"continual learning" AND (abs:theory OR abs:guarantee OR abs:bound OR abs:provable)',
             oa='title_and_abstract.search:("exemplar-free" OR "rehearsal-free" OR "replay-free" OR "memory-free"),title_and_abstract.search:"continual learning",title_and_abstract.search:(theory OR guarantee OR bound OR provable)'),
 "S14": dict(ax='abs:"continual learning" AND (abs:provable OR abs:provably OR abs:guarantee OR abs:guarantees) AND abs:forgetting',
             oa='title_and_abstract.search:"continual learning",title_and_abstract.search:(provable OR provably OR guarantee OR guarantees),title_and_abstract.search:forgetting'),
 "S15": dict(ax='(abs:"delta rule" OR abs:DeltaNet OR abs:"gated delta" OR abs:"fast weight" OR abs:"fast weights" OR abs:"test-time training" OR abs:"test-time regression" OR abs:"linear attention") AND (abs:theory OR abs:theoretical OR abs:convergence OR abs:capacity OR abs:interference OR abs:forgetting OR abs:retention OR abs:Kaczmarz OR abs:"least squares" OR abs:regret OR abs:bound)',
             oa='title_and_abstract.search:("delta rule" OR DeltaNet OR "gated delta" OR "fast weight" OR "fast weights" OR "test-time training" OR "test-time regression" OR "linear attention"),title_and_abstract.search:(theory OR theoretical OR convergence OR capacity OR interference OR forgetting OR retention OR Kaczmarz OR "least squares" OR regret OR bound)'),
 "S16": dict(ax='(abs:cyclic OR abs:"task ordering" OR abs:"random ordering" OR abs:"random orderings" OR abs:"task order") AND (abs:"continual learning" OR abs:"catastrophic forgetting" OR abs:forgetting)',
             oa='title_and_abstract.search:(cyclic OR "task ordering" OR "random ordering" OR "random orderings" OR "task order"),title_and_abstract.search:("continual learning" OR "catastrophic forgetting" OR forgetting)'),
 "S17": dict(ax='(abs:"online learner" OR abs:"online learners" OR abs:"online optimization" OR abs:"online regression") AND (abs:"state space" OR abs:"sequence model" OR abs:"sequence models" OR abs:recurrent OR abs:"associative memory")',
             oa='title_and_abstract.search:("online learner" OR "online learners" OR "online optimization" OR "online regression"),title_and_abstract.search:("state space" OR "sequence model" OR "sequence models" OR recurrent OR "associative memory")'),
}
CAP = 2500
log = {}


def get(url, tries=4):
    for k in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()
        except Exception as e:
            time.sleep(5 * (k + 1)); err = e
    raise err


def ax_fetch(q, extra=""):
    NS = {"a": "http://www.w3.org/2005/Atom", "o": "http://a9.com/-/spec/opensearch/1.1/"}
    recs, start, total = [], 0, None
    while True:
        url = "https://export.arxiv.org/api/query?search_query=" + urllib.parse.quote(q + extra) + f"&start={start}&max_results=200&sortBy=submittedDate&sortOrder=descending"
        root = ET.fromstring(get(url)); time.sleep(3.5)
        total = int(root.find("o:totalResults", NS).text); ents = root.findall("a:entry", NS)
        for e in ents:
            i = e.find("a:id", NS).text; aid = re.sub(r"v\d+$", "", i.split("/abs/")[-1])
            recs.append(dict(src="arxiv", id=aid, arxiv=aid, doi=None, title=" ".join(e.find("a:title", NS).text.split()), year=int(e.find("a:published", NS).text[:4]),
                             abstract=" ".join((e.find("a:summary", NS).text or "").split()), venue="arXiv", cited=None))
        start += 200
        if not ents or start >= min(total, CAP): break
    return recs, total


def ax_all(q):
    recs, total = ax_fetch(q)
    if total <= CAP: return recs, total, False
    out, tot_all, trunc = [], 0, False
    for y in range(2014, 2027):
        r, t = ax_fetch(q, f" AND submittedDate:[{y}01010000 TO {y}12312359]"); out += r; tot_all += t
        if t > CAP: trunc = True
    return out, total, trunc


def oa_abs(inv):
    if not inv: return ""
    pos = {}
    for w, ps in inv.items():
        for p in ps: pos[p] = w
    return " ".join(pos[i] for i in sorted(pos))


def oa_all(flt):
    recs, cursor, total = [], "*", None
    while cursor and len(recs) < CAP:
        url = "https://api.openalex.org/works?filter=" + urllib.parse.quote(flt, safe=":,()") + "&per-page=200&cursor=" + urllib.parse.quote(cursor) + "&select=id,doi,title,publication_year,abstract_inverted_index,cited_by_count,primary_location,ids,type"
        j = json.loads(get(url)); total = j["meta"]["count"]; time.sleep(0.3)
        for w in j["results"]:
            ids = w.get("ids") or {}; ar = None
            m = re.search(r"arxiv\.org/abs/([\w./-]+)", ids.get("arxiv", "") or "")
            if m: ar = re.sub(r"v\d+$", "", m.group(1))
            elif w.get("doi") and "arxiv." in w["doi"].lower(): ar = w["doi"].lower().split("arxiv.")[-1]
            loc = (w.get("primary_location") or {}).get("source") or {}
            recs.append(dict(src="openalex", id=w["id"], arxiv=ar, doi=w.get("doi"), title=" ".join((w.get("title") or "").split()), year=w.get("publication_year"), abstract=oa_abs(w.get("abstract_inverted_index")), venue=loc.get("display_name") or "", cited=w.get("cited_by_count"), type=w.get("type")))
        cursor = j["meta"].get("next_cursor")
        if not j["results"]: break
    return recs, total, total > CAP


if __name__ == "__main__":
    only = sys.argv[1:] or list(QUERIES)
    for qid in only:
        d = QUERIES[qid]
        for src in ("arxiv", "openalex"):
            fn = f"{RAW}/{src}_{qid}.json"
            if os.path.exists(fn): continue
            try:
                recs, total, trunc = ax_all(d["ax"]) if src == "arxiv" else oa_all(d["oa"])
            except Exception as e:
                print(qid, src, "FAILED", repr(e), flush=True); log[f"{src}_{qid}"] = dict(error=repr(e)); continue
            json.dump(recs, open(fn, "w")); log[f"{src}_{qid}"] = dict(total=total, fetched=len(recs), truncated=trunc)
            print(qid, src, "total", total, "fetched", len(recs), "TRUNCATED" if trunc else "", flush=True)
    old = json.load(open(HERE + "/search_log.json")) if os.path.exists(HERE + "/search_log.json") else {}
    old.update(log); json.dump(old, open(HERE + "/search_log.json", "w"), indent=1)
