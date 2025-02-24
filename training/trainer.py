import torch

def train_model(model, dataloader, optimizer, loss_fn, device):
    model.train()
    for batch in dataloader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)  # Shape: (batch_size, seq_len, vocab_size)

        # Reshape targets to match CrossEntropyLoss expectations
        targets = targets.view(-1)  # Flatten to (batch_size * seq_len,)
        outputs = outputs.view(-1, outputs.size(-1))  # Reshape to (batch_size * seq_len, vocab_size)

        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
