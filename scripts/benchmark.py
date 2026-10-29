"""Compare equal-ranking scan/index retrieval, including optional NLP latency."""
import argparse
import json
import statistics
import time
import torch
from workflows.retrieval import Retriever, ROOT
from workflows.nlp import Normalizer


def measure(fn, queries, rounds):
    for q in queries:
        fn(q)
    values = []
    for _ in range(rounds):
        for q in queries:
            start = time.perf_counter_ns()
            fn(q)
            values.append((time.perf_counter_ns() - start)/1e6)
    return {"median_ms": statistics.median(values), "p95_ms": sorted(values)[int(0.95*(len(values)-1))]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--copies", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=20)
    args = parser.parse_args()
    if args.copies < 1 or args.rounds < 1:
        parser.error("copies and rounds must be positive")
    torch.set_num_threads(1)
    base = Retriever().documents
    docs = [{**d, "id": f'{d["id"]}-{i}'} for i in range(args.copies) for d in base]
    retriever = Retriever(docs)
    nlp = Normalizer()
    queries = ["explain mcp tool integration", "what is document retrieval index", "how does flask model serving work", "explain model evaluation latency"]
    assert all(retriever.search(q, indexed=False) == retriever.search(q) for q in queries)
    baseline = measure(lambda q: retriever.search(q, indexed=False), queries, args.rounds)
    indexed = measure(retriever.search, queries, args.rounds)
    pipeline = measure(lambda q: retriever.search(nlp.normalize(q)["query"]), queries, args.rounds)
    labels = ["mcp", "retrieval", "flask", "evaluation"]
    def hit_rate(rewrite):
        return sum(any(d["id"].rsplit("-", 1)[0] == label for d in retriever.search(nlp.normalize(q)["query"] if rewrite else q)) for q, label in zip(queries, labels))/len(queries)
    result = {"corpus_size": len(docs), "corpus": "synthetic repeated documents", "rounds": args.rounds, "queries": len(queries), "scan": baseline, "index": indexed, "normalize_plus_index": pipeline, "model_loaded": nlp.model is not None, "index_speedup_percent": 100*(1-indexed["median_ms"]/baseline["median_ms"]), "end_to_end_speedup_percent": 100*(1-pipeline["median_ms"]/baseline["median_ms"]), "ranking_equivalence": True, "scope": "In-process retrieval; excludes MCP, HTTP and LLM overhead. No production claim."}
    result["topic_hit_at_3_original"] = hit_rate(False)
    result["topic_hit_at_3_rewritten"] = hit_rate(True)
    result["quality_scope"] = "Four labeled synthetic queries; topic hit rate, not document recall on real data"
    (ROOT / "artifacts/benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
