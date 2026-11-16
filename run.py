"""retrieval-eval: recall@k and MRR over golden query-doc pairs.

Each case lists the relevant doc (plus acceptable alternates). A run passes when every
query surfaces a relevant doc inside k and the mean reciprocal rank clears the gate.
Exit 1 = eval blocks the retriever.
"""
import json, sys

def load_retriever(path):
    if not path:
        return demo_retriever
    import importlib.util
    s = importlib.util.spec_from_file_location("retriever", path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m.run

DOCS = {}

def demo_retriever(query, k):
    """Naive retriever: first-word overlap only - misses paraphrases."""
    q = query.lower().split()
    scored = sorted(DOCS.items(), key=lambda kv: -sum(1 for w in q[:3] if w in kv[1]))
    return [d for d, _ in scored[:k]]

def main():
    global DOCS
    spec = json.load(open("cases.json"))
    DOCS = spec["docs"]
    retr = load_retriever(sys.argv[1] if len(sys.argv) > 1 else None)
    k = spec["k"]
    recalls, rrs = [], []
    for c in spec["cases"]:
        ranked = retr(c["query"], k)
        acceptable = set(c["relevant"]) | set(c.get("also_ok", []))
        hit_rank = next((i + 1 for i, d in enumerate(ranked[:k]) if d in acceptable), None)
        recalls.append(1 if hit_rank else 0)
        rrs.append(1 / hit_rank if hit_rank else 0)
        print(f"[{c['id']}] {'hit @%d' % hit_rank if hit_rank else 'MISS'} (top-{k}: {ranked[:k]})")
    recall = sum(recalls) / len(recalls)
    mrr = sum(rrs) / len(rrs)
    g = spec["gates"]
    ok = recall >= g["min_recall_at_k"] and mrr >= g["min_mrr"]
    print(f"\nrecall@{k} {recall:.2f} (gate {g['min_recall_at_k']}), MRR {mrr:.2f} (gate {g['min_mrr']})")
    print("RETRIEVER:", "PASS" if ok else "BLOCKED")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
