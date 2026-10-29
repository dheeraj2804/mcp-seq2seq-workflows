"""Small word-level GRU encoder-decoder. No checkpoint means identity fallback."""
from pathlib import Path
import threading
import torch
from torch import nn
from .retrieval import tokens, ROOT

SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, hidden=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.encoder = nn.GRU(hidden, hidden, batch_first=True)
        self.decoder = nn.GRU(hidden, hidden, batch_first=True)
        self.output = nn.Linear(hidden, vocab_size)

    def forward(self, source, target):
        _, state = self.encoder(self.embedding(source))
        decoded, _ = self.decoder(self.embedding(target), state)
        return self.output(decoded)

    def generate(self, source, max_tokens=12):
        _, state = self.encoder(self.embedding(source))
        token = torch.tensor([[1]])
        result = []
        for _ in range(max_tokens):
            decoded, state = self.decoder(self.embedding(token), state)
            token = self.output(decoded[:, -1:]).argmax(-1)
            value = token.item()
            if value == 2:
                break
            if value >= 4:
                result.append(value)
        return result

class Normalizer:
    def __init__(self, checkpoint=None):
        self.model = None
        self.lock = threading.Lock()
        path = Path(checkpoint) if checkpoint else ROOT / "artifacts/seq2seq.pt"
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=True)
            self.vocab = saved["vocab"]
            self.lookup = {w: i for i, w in enumerate(self.vocab)}
            self.model = Seq2Seq(len(self.vocab), saved["hidden"])
            self.model.load_state_dict(saved["state_dict"])
            self.model.eval()

    def normalize(self, text):
        if self.model is None:
            return {"query": text, "mode": "identity", "reason": "No trained checkpoint"}
        words = tokens(text)[:48]
        # Preserve information on out-of-domain input rather than hallucinate a rewrite.
        if not words or any(w not in self.lookup for w in words):
            return {"query": text, "mode": "identity", "reason": "Out-of-vocabulary input"}
        source = torch.tensor([[self.lookup[w] for w in words] + [2]])
        with self.lock, torch.inference_mode():
            output = self.model.generate(source)
        query = " ".join(self.vocab[i] for i in output)
        return {"query": query or text, "mode": "seq2seq" if query else "identity"}
