"""
This implementation is completely based on the version used in TabSyn. 
The Stage 2 component and the train2 code are identical to the original 

Reference:
[1] https://github.com/amazon-science/tabsyn/blob/main/tabsyn/main.py
"""

import os
import torch
import wandb
import numpy as np

from torch.utils.data import DataLoader, TensorDataset

def train_function(config, model, train_z, llm_emb, optimizer, scheduler, device):
    # llm_emb: 각 train_z 행에 대응하는 LLM 텍스트 임베딩 [N, llm_dim] (None이면 무조건부 학습)
    base_name = f"{config['dataset']}_{config['lr1']}_{config['d_token']}_{config['denoising_dim']}"
    base_name += f"_{config['batch_size1']}_{config['batch_size2']}_{config['max_beta']}"
    base_name += f"_{config['var']}_{config['lambda']}"
    model_dir = f"./assets/models/{base_name}/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_name = f"{base_name}_{config['seed']}"

    mean = train_z.mean(0)
    train_z = (train_z - mean) / 2
    
    num_epochs = config['epochs_2']

    # (z, emb) 쌍이 같은 행끼리 배치로 묶이도록 TensorDataset 사용
    if llm_emb is None:
        llm_emb = torch.zeros(train_z.shape[0], 0)
    llm_emb = torch.as_tensor(llm_emb).float()
    assert llm_emb.shape[0] == train_z.shape[0], "llm_emb와 train_z의 행 수가 다릅니다"
    use_cond = llm_emb.shape[1] > 0

    train_dataloader = DataLoader(
        TensorDataset(train_z, llm_emb),
        batch_size = config["batch_size2"],
        shuffle = True,
        num_workers = 4,
    )
    print("train_z:", tuple(train_z.shape), "llm_emb:", tuple(llm_emb.shape))
    best_loss = float('inf')
    patience = 0
    for epoch in range(num_epochs):
        logs = {
            'diffusion_loss' : [],
        }
        batch_loss = 0.0
        len_input = 0
        for batch, emb_batch in train_dataloader:
            inputs = batch.float().to(device)
            emb_batch = emb_batch.to(device) if use_cond else None
            loss = model(inputs, emb_batch)
        
            loss = loss.mean()

            batch_loss += loss.item() * len(inputs)
            len_input += len(inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        curr_loss = batch_loss / len_input
        scheduler.step(curr_loss)
        logs["diffusion_loss"].append(curr_loss)
        if curr_loss < best_loss:
            best_loss = curr_loss
            patience = 0
            torch.save(model.state_dict(), f"./{model_dir}/stage2_{model_name}.pth")  
        else:
            patience += 1
            if patience == 500:
                print('Early stopping')
                break
        if epoch % 1000 == 0:
            torch.save(model.state_dict(), f"./{model_dir}/stage2_{model_name}.pth")  
        
        if epoch % 500 == 0 or epoch == 1:
            print(f"[Epoch {epoch:4d}/{num_epochs}] diffusion_loss: {curr_loss:.6f}")
        
        wandb.log({x : np.mean(y) for x, y in logs.items()})