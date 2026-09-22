"""Reproduce every SynthPAI number quoted in docs/methodology-review.md.

    pip install huggingface_hub
    python scripts/dataset_report.py [path/to/synthpai.jsonl]

With no argument it downloads the pinned revision. Prints aggregates only --
no comment text -- so the output is safe to paste into a report or a PR.

This is a read-only research script. It is deliberately standalone and does
not import from src/: nothing here is part of the product yet.
"""

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

DATASET_ID = "RobinSta/SynthPAI"
REVISION = "b572595f543a51db789caddbb81a9fc4edc6c32f"

# our name -> SynthPAI profile / review field name
ATTRS = {
    "age": "age",
    "location": "city_country",
    "education": "education",
    "occupation": "occupation",
    "income": "income_level",
}

_PHD = re.compile(r"\b(phd|ph\.d|doctorate|doctoral)\b", re.I)
_MASTER = re.compile(r"\b(master|masters|msc|m\.sc|mba|m\.a|meng)\b", re.I)
_COLLEGE = re.compile(r"\b(bachelor|bachelors|bsc|b\.sc|b\.a|ba|beng|college|undergraduate|degree)\b", re.I)


def education_level(text):
    """SynthPAI education is free text (206 distinct values / 300 profiles).
    Map to the four-way scale the SynthPAI paper reports on, so our numbers
    are comparable to its published baselines."""
    if not text:
        return None
    if _PHD.search(text):
        return "phd"
    if _MASTER.search(text):
        return "master"
    if _COLLEGE.search(text):
        return "college"
    return "high_school"


def age_bucket(age):
    if not isinstance(age, (int, float)):
        return None
    for name, lo, hi in [("18-24", 18, 24), ("25-34", 25, 34), ("35-44", 35, 44),
                         ("45-54", 45, 54), ("55+", 55, 200)]:
        if lo <= age <= hi:
            return name
    return None


def country_of(city_country):
    if not city_country:
        return None
    parts = [p.strip() for p in city_country.split(",") if p.strip()]
    return parts[-1] if parts else None


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def download():
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=DATASET_ID, filename="synthpai.jsonl",
                           repo_type="dataset", revision=REVISION,
                           local_dir="data/raw")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else download()
    rows = load(path)
    profiles = {r["author"]: r["profile"] for r in rows}

    print(f"revision  {REVISION}")
    print(f"comments  {len(rows)}   authors {len(profiles)}"
          f"   threads {len({r.get('thread_id') for r in rows})}\n")

    # --- inferability: is the attribute inferable from THIS comment? --------
    # A non-empty human review estimate is the label. Profile attributes
    # attach to the author, not the comment, so they cannot serve this role.
    print("INFERABILITY (human review estimate present on this comment)")
    counts = Counter()
    for r in rows:
        human = (r.get("reviews") or {}).get("human") or {}
        for ours, theirs in ATTRS.items():
            if str((human.get(theirs) or {}).get("estimate", "")).strip():
                counts[ours] += 1
    for a, n in counts.most_common():
        print(f"  {a:<12}{n:>6}{100 * n / len(rows):>8.1f}%")

    # --- hardness: what kind of cue carried the inference -------------------
    print("\nPUBLISHED GUESS HARDNESS")
    hard = Counter(g.get("hardness") for r in rows for g in (r.get("guesses") or [])
                   if g.get("feature") in ATTRS.values())
    tot = sum(hard.values()) or 1
    for h, n in hard.most_common():
        print(f"  {h:<14}{n:>6}{100 * n / tot:>8.1f}%")
    print(f"  -> {100 * (tot - hard['direct']) / tot:.1f}% of inferences are NOT "
          f"from an explicit mention")

    # --- attack accuracy: model_eval is TERNARY (0 / 0.5 / 1) ---------------
    print("\nPUBLISHED ATTACK ACCURACY (reference baseline)")
    print(f"  {'attribute':<12}{'n':>6}{'top1':>9}{'top1+part':>11}{'top3':>8}")
    graded = {a: [] for a in ATTRS}
    for r in rows:
        for g in (r.get("guesses") or []):
            for ours, theirs in ATTRS.items():
                if g.get("feature") == theirs and g.get("model_eval"):
                    graded[ours].append([float(v) for v in g["model_eval"]])
    for a, evs in graded.items():
        if not evs:
            continue
        n = len(evs)
        s1 = sum(e[0] == 1 for e in evs)
        l1 = sum(e[0] >= 0.5 for e in evs)
        s3 = sum(any(v == 1 for v in e[:3]) for e in evs)
        print(f"  {a:<12}{n:>6}{100*s1/n:>8.1f}%{100*l1/n:>10.1f}%{100*s3/n:>7.1f}%")
    print("  -> top1 vs top1+part is partial credit alone. Any before/after")
    print("     comparison must fix one convention and say which.")
    print("  -> top3 is near-100% for age and income because those attributes")
    print("     have few values; 3 guesses nearly cover the space. Not a useful metric there.")

    # --- does the frozen taxonomy fit the data? -----------------------------
    print("\nPROFILE GROUND TRUTH (300 profiles)")
    ctry = Counter(country_of(p.get("city_country")) for p in profiles.values())
    ctry.pop(None, None)
    usa = ctry.get("USA", 0) + ctry.get("United States", 0)
    print(f"  countries {len(ctry)}   USA {usa}/{len(profiles)} ({100*usa/len(profiles):.0f}%)"
          f"   top: " + ", ".join(f"{k} {v}" for k, v in ctry.most_common(5)))
    print("  -> artifacts/taxonomy.json has 4 US-region classes covering 7% of profiles")
    edu = Counter(education_level(p.get("education")) for p in profiles.values())
    print("  education " + "  ".join(f"{k} {100*v/len(profiles):.1f}%" for k, v in edu.most_common()))
    ages = Counter(age_bucket(p.get("age")) for p in profiles.values())
    print("  age       " + "  ".join(f"{k} {100*ages[k]/len(profiles):.1f}%"
                                     for k in ["18-24", "25-34", "35-44", "45-54", "55+"]))
    nums = [p["age"] for p in profiles.values() if isinstance(p.get("age"), int)]
    print(f"  -> median age {sorted(nums)[len(nums)//2]}; the demo persona (undergraduate) "
          f"is the smallest bucket")

    # --- split sanity: sorted() before shuffle, see issue #21 ---------------
    print("\nPROFILE-DISJOINT SPLIT (seed 42, authors sorted before shuffle)")
    authors = sorted(profiles)
    random.Random(42).shuffle(authors)
    n_tr, n_va = round(len(authors) * 0.70), round(len(authors) * 0.15)
    split = {"train": authors[:n_tr], "val": authors[n_tr:n_tr+n_va], "test": authors[n_tr+n_va:]}
    for name, ids in split.items():
        sub = [r for r in rows if r["author"] in set(ids)]
        loc = sum(1 for r in sub
                  if str((((r.get("reviews") or {}).get("human") or {}).get("city_country") or {})
                         .get("estimate", "")).strip())
        print(f"  {name:<6} authors {len(ids):>4}  comments {len(sub):>5}  location positives {loc:>4}")
    print("  -> report a confidence interval with any location claim")


if __name__ == "__main__":
    main()
