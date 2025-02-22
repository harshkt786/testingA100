import torch
import time
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

def time_operation(operation, *args, **kwargs):
    start = time.time()
    result = operation(*args, **kwargs)
    end = time.time()
    return end - start, result

def benchmark_matrix_multiply(size):
    a = torch.randn(size, size, device='cuda')
    b = torch.randn(size, size, device='cuda')
    return time_operation(torch.matmul, a, b)

def benchmark_transformer_layer(seq_length, batch_size, d_model):
    layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8).cuda()
    x = torch.randn(seq_length, batch_size, d_model, device='cuda')
    return time_operation(layer, x)

class SimpleTransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(50000, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers, num_layers, batch_first=True)
        self.fc = nn.Linear(d_model, 50000)
    
    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        output = self.transformer(src, tgt)
        return self.fc(output)

def benchmark_training_step(model, optimizer, criterion, src, tgt):
    def training_step():
        model.zero_grad()
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        output = model(src, tgt_input)
        loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()
    return time_operation(training_step)

def create_dummy_data(vocab_size, seq_length, batch_size):
    src = torch.randint(0, vocab_size, (batch_size, seq_length), device='cuda')
    tgt = torch.randint(0, vocab_size, (batch_size, seq_length), device='cuda')
    return src, tgt

def benchmark_llm_training(vocab_size, d_model, seq_length, batch_size, num_epochs):
    model = SimpleTransformerModel(d_model, 8, 6).cuda()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    src, tgt = create_dummy_data(vocab_size, seq_length, batch_size)
    dataset = TensorDataset(src, tgt)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    total_time = 0
    for epoch in range(num_epochs):
        for batch_src, batch_tgt in dataloader:
            time_taken, _ = benchmark_training_step(model, optimizer, criterion, batch_src, batch_tgt)
            total_time += time_taken
    return total_time / (num_epochs * len(dataloader))

def main():
    print("A100 GPU LLM Benchmark")
    
    print("\n1. Matrix Multiplication Benchmark:")
    for size in [1000, 5000, 10000]:
        time_taken, _ = benchmark_matrix_multiply(size)
        print(f"  {size}x{size} matrix multiplication: {time_taken:.4f} seconds")
    
    print("\n2. Transformer Layer Benchmark:")
    for seq_length in [512, 1024, 2048]:
        time_taken, _ = benchmark_transformer_layer(seq_length, batch_size=32, d_model=768)
        print(f"  Sequence length {seq_length}: {time_taken:.4f} seconds")
    
    print("\n3. LLM Training Benchmark:")
    vocab_size = 50000
    d_model = 768
    seq_length = 512
    batch_size = 32
    num_epochs = 3
    avg_time = benchmark_llm_training(vocab_size, d_model, seq_length, batch_size, num_epochs)
    print(f"  Average training step time: {avg_time:.4f} seconds")
    
    print(f"\nEstimated time for 1000 training steps: {avg_time * 1000:.2f} seconds")

if __name__ == "__main__":
    main()
