import torch

def train_model(model, dataloader, optimizer, loss_fn, device, vocab):
    model.train()
    for batch in dataloader:
        inputs, targets = batch  # `inputs` is raw text
        inputs, targets = list(inputs), targets.to(device)  

        # Tokenize & convert text to token IDs
        token_ids = [vocab.numericalize(text) for text in inputs]

        # Convert to tensor and pad sequences
        max_len = max(len(seq) for seq in token_ids)
        token_ids = [seq + [0] * (max_len - len(seq)) for seq in token_ids]  # Pad with <pad> token
        token_ids = torch.tensor(token_ids, dtype=torch.long).to(device)

        optimizer.zero_grad()
        outputs = model(token_ids)  

        targets = targets.view(-1)  
        outputs = outputs.view(-1, outputs.size(-1))  

        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
