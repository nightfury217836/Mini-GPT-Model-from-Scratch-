import torch
import torch.nn as nn
import sys
import os

# -------- Load Model ---------
checkpoint = torch.load("poem_gpt.pth", map_location="cpu")

stoi = checkpoint["stoi"]
itos = checkpoint["itos"]
vocab_size = checkpoint["vocab_size"]

seq_len = 12
embed_dim = 64
num_heads = 4
num_layers = 2

class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = nn.Embedding(seq_len, embed_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        positions = torch.arange(0, x.size(1)).unsqueeze(0)
        x = self.embed(x) + self.pos(positions)
        x = self.transformer(x)
        logits = self.fc(x)
        return logits
    

model = MiniGPT()
model.load_state_dict(checkpoint["model_state"])
model.eval()


# -------- Helper ---------
def encode(words):
    return torch.tensor([stoi.get(w, 0) for w in words], dtype=torch.long)

def decode(ids):
    return " ".join([itos[i] for i in ids])


# -------------Text Generation -----------
def generate(start, max_new_words=10):
    tokens = start.lower().split()
    input_ids = encode(tokens).unsqueeze(0)

    for _ in range(max_new_words):
        if input_ids.size(1) > seq_len:
            input_ids = input_ids[:, -seq_len:]

        logits = model(input_ids)
        next_token_logits = logits[0, -1]
        probs = torch.softmax(next_token_logits, dim=0)
        next_id = torch.multinomial(probs, 1). item()

        tokens.append(itos[next_id])
        input_ids = torch.tensor([[stoi[w] for w in tokens]])

    return " ".join(tokens)

# ---------- Terminal Interface -------------

def clear():
    os.system("cls" if os.name == "nt" else "clear")

clear()
print("────────────────────────────────────────")
print("  🌙✨ Mini GPT Poem Generator ✨🌙")
print("────────────────────────────────────────")

print("Type a starting phrase to get a one-line poem.")
print("Type 'quit' to exit.\n")


while True:
    start = input("💬 Enter a starting word/phrase: ").strip()
    if start.lower() == "quit":
        print("\n👋 Exiting... Goodbye!\n")
        sys.exit()

    poem = generate(start, max_new_words=10)
    poem = poem[0].upper() + poem[1:]
    if poem[-1] not in ".!?":
        poem += "."

    print("\n📜  Your Generated Poem:")
    print("   \"" + poem + "\"\n")   # double quotes around output
