"""CPU-friendly synthetic demonstration; not a production language model."""
import argparse
import json
import random
import torch
from workflows.nlp import Seq2Seq, SPECIAL, Normalizer
from workflows.retrieval import ROOT, tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    random.seed(42)
    intents = ["mcp tool integration", "workflow retry timeout", "seq2seq query normalization", "document retrieval index", "flask model serving", "claude openai orchestration", "model evaluation latency"]
    templates = ["explain {}", "how does {} work", "tell me about {}", "what is {}", "help me understand {}", "describe {}"]
    # Split by question template before training. The vocabulary may contain held-out
    # words; no held-out target sequence is passed to the optimizer.
    train = [(t.format(i), i) for t in templates[:4] for i in intents]
    test = [(t.format(i), i) for t in templates[4:] for i in intents]
    vocab = SPECIAL + sorted({w for a, b in train + test for w in tokens(a + " " + b)})
    lookup = {w: i for i, w in enumerate(vocab)}
    model = Seq2Seq(len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    for epoch in range(args.epochs):
        random.shuffle(train)
        total = 0
        for source, target in train:
            x = torch.tensor([[lookup[w] for w in tokens(source)] + [2]])
            y = [1] + [lookup[w] for w in tokens(target)] + [2]
            optimizer.zero_grad()
            loss = criterion(model(x, torch.tensor([y[:-1]])).reshape(-1, len(vocab)), torch.tensor(y[1:]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        if epoch == 0 or (epoch+1) % 20 == 0:
            print(f"epoch={epoch+1} loss={total/len(train):.4f}")
    path = ROOT / "artifacts/seq2seq.pt"
    path.parent.mkdir(exist_ok=True)
    torch.save({"vocab": vocab, "hidden": 64, "state_dict": model.state_dict()}, path)
    normalizer = Normalizer(path)
    examples = [{"input": a, "expected": b, "actual": normalizer.normalize(a)["query"]} for a,b in test]
    metrics = {"seed": 42, "train_examples": len(train), "test_examples": len(test), "split": "held-out templates; synthetic shared intents", "exact_match": sum(e["expected"] == e["actual"] for e in examples)/len(examples), "examples": examples}
    (ROOT / "artifacts/training_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({k:v for k,v in metrics.items() if k != "examples"}, indent=2))

if __name__ == "__main__":
    main()
