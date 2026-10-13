"""Key-value recall prompts for the downstream certificate test.  Torch/numpy-free (random + tokenizer only)."""
import random
NAMES = "Alice Bob Carol David Emma Frank Grace Henry Irene Jack Karen Leo Maria Nick Olivia Paul Quinn Rachel Steve Tina Uma Victor Wendy Xavier Yara Zack Adam Bella Chris Diana Ethan Fiona George Hannah Ivan Julia Kevin Laura Mike Nora Oscar Peter Rita Sam Tom Ursula Vera Will Zoe Anna Ben Cara Dan Eve Finn Gina Hugo Ian Jane Kyle Lisa Mark Nina Owen Pam Ray Sara Todd Una Vince Walt Yuri Abby Brad Cindy Drew Elle Fred Gwen Hank Iris Jake Kate Liam Mona Ned Opal Pete Rose Seth Tess Uri Val Wes Xena Yvonne Zane".split()
WORDS = "red blue green yellow purple orange pink brown black white gray silver gold apple bread cheese milk water rice salt sugar honey tea coffee cat dog bird fish horse cow pig sheep goat duck wolf bear lion tiger fox deer frog snake whale shark mouse rabbit tree flower grass river lake sea hill rock sand rain snow wind fire ice cloud star moon sun sky road door window chair table bed lamp book pen cup plate knife fork spoon bottle box bag hat shoe coat shirt ring watch clock key map coin card ticket letter song story game dream city town farm school garden bridge tower castle ship train bus car bike plane boat".split()


def build_vocab(tok):
    names, words = [], []
    for w in NAMES:
        ids = tok(" " + w, add_special_tokens=False)["input_ids"]
        if len(ids) == 1: names.append((w, ids[0]))
    for w in WORDS:
        ids = tok(" " + w, add_special_tokens=False)["input_ids"]
        if len(ids) == 1: words.append((w, ids[0]))
    is_id = tok(" is", add_special_tokens=False)["input_ids"]; dot_id = tok(".", add_special_tokens=False)["input_ids"]
    assert len(is_id) == 1 and len(dot_id) == 1, (is_id, dot_id)
    return names, words, is_id[0], dot_id[0]


def make_prompt(rng, vocab, n, bos):
    names, words, is_id, dot_id = vocab
    ks = rng.sample(names, n); vs = rng.sample(words, n)
    ids, spans = [bos], []
    for (k, kid), (v, vid) in zip(ks, vs):
        s = len(ids); ids += [kid, is_id, vid, dot_id]; spans.append((s, s + 3))       # span = key, "is", value
    q = rng.randrange(n); qpos = len(ids)
    ids += [ks[q][1], is_id]
    return dict(ids=ids, spans=spans, q=q, target=vs[q][1], qpos=qpos, n=n, age=qpos - spans[q][1], cands=[v[1] for v in vs])
