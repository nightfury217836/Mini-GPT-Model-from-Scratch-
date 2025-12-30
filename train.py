import torch
import torch.nn as nn
import torch.optim as optim

# ---- Load Dataset ----
with open("data/poems.txt", "r", encoding="utf8") as f:
    text = f.read().lower()

# ---- Word Tokenizer ----
words = text.split()
vocab = sorted(list(set(words)))
vocab_size = len(vocab)

stoi = {w: i for i, w in enumerate(vocab)}
itos = {i: w for w, i in stoi.items()}

def encode(words):  # list of word tokens → ids
    return torch.tensor([stoi[w] for w in words], dtype=torch.long)

def decode(ids):  # ids → string
    return " ".join([itos[i] for i in ids])


# ---- Hyperparameters ----
seq_len = 12
embed_dim = 64
num_heads = 4
num_layers = 2
lr = 0.002
epochs = 2000

# ---- Prepare Training Data ----
tokens = encode(words)

def get_batch():
    idx = torch.randint(0, len(tokens) - seq_len - 1, (1,))
    x = tokens[idx : idx + seq_len]
    y = tokens[idx + 1 : idx + seq_len + 1]
    return x.unsqueeze(0), y.unsqueeze(0)


# ---- GPT-Style Transformer ----
class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = nn.Embedding(seq_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        positions = torch.arange(0, x.size(1)).unsqueeze(0)
        x = self.embed(x) + self.pos(positions)
        x = self.transformer(x)
        logits = self.fc(x)
        return logits


model = MiniGPT()
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()


# ---- Training Loop ----
for epoch in range(epochs):
    x, y = get_batch()
    logits = model(x)
    loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# ---- Save model ----
torch.save({
    "model_state": model.state_dict(),
    "stoi": stoi,
    "itos": itos,
    "vocab_size": vocab_size
}, "poem_gpt.pth")

print("\nTraining complete. Model saved as poem_gpt.pth")
