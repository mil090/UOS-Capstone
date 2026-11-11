#%%
import os
# LLM 토크나이저 사용 후 DataLoader worker를 fork할 때 나오는 경고/교착 방지
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import argparse
import importlib
import sys

# 작업 디렉토리를 이 스크립트가 있는 위치(uoscapstone/SPT/SPT)로 이동
target_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() \
    else '/gpfs/home1/mil090/uoscapstone/SPT/SPT'
os.chdir(target_dir)
if target_dir not in sys.path:
    sys.path.insert(0, target_dir)

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from modules.utils import set_random_seed

import sys
import subprocess
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("./wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb

project = "uos-capstone" # put your WANDB project name ### tabsyn_ours
entity = "mil090-university-of-seoul" # put your WANDB username

run = wandb.init(
    project=project, 
    entity=entity, 
    tags=["train"], # put tags of this python project
)
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--seed', type=int, default=0, 
                        help='seed for repeatable results')
    parser.add_argument("--model", type=str, default="SPT")
    parser.add_argument('--dataset', type=str, default='anuran', 
                        help="""
                        Tabular dataset options: 
                        abalone, adult, anuran, banknote, breast, concrete,
                        kings, letter, loan, redwine, shoppers, whitewine
                        """)
    parser.add_argument('--test_size', default=0.2, type=float, 
                        help="Proportion of the dataset to include in the test split")

    ### stage-1 model
    parser.add_argument('--batch_size1', default=512, type=int, 
                        help="Batch size for stage 1 training")
    parser.add_argument('--var', default=0.1, type=float,
                        help='The value of fixed encoder variance') 
    parser.add_argument('--lr1', default=0.001, type=float,
                        help='Learning rate for stage 1 training')
    parser.add_argument('--weight_decay1', default=0, type=float,
                        help='weight decay for stage 1')
    parser.add_argument('--d_token', default=4, type=int,
                        help='Latent dimension')
    parser.add_argument('--num_layers', default=2, type=int,
                        help='The number of layer in transformer')
    parser.add_argument('--epochs', default=1, type=int,
                        help='Training epochs for stage 1') 
    parser.add_argument('--max_beta', type=float, default=1e-2, 
                        help='Maximum value of regularization weight')
    parser.add_argument('--min_beta', type=float, default=1e-5, 
                        help='Manimum value of regularization weight')
    parser.add_argument('--lambda', type=float, default=0.7, 
                        help='Initial value of regularization weight')
    parser.add_argument('--factor', default=32, type=int,
                        help='FACTOR')
    parser.add_argument('--n_head', default=1, type=int,
                        help='N_HEAD')
    parser.add_argument('--bias', default=True, type=bool,
                        help='Token Bias')        
    ### stage-2 Model
    parser.add_argument('--batch_size2', default=512, type=int,
                        help='Batch size for stage 2 training' )
    parser.add_argument('--scheduler', type=str, default='linear', 
                        help="Options for beta scheduling: linear and cosine")
    parser.add_argument('--weight_decay2', default=1e-4, type=float,
                        help='Weight decay for AdamW in training stage 2')
    parser.add_argument('--epochs_2', default=10_000, type=int,
                        help='Training epochs for stage 2')
    parser.add_argument("--denoising_dim", default=1024, type=int,
                        help="Size of latent dimension for stage 2")
    ### LLM 조건 임베딩
    parser.add_argument("--llm_model", default="Qwen/Qwen2.5-0.5B-Instruct", type=str,
                        help='HuggingFace decoder LLM used to embed text conditions')
    parser.add_argument("--reuse_stage1", action="store_true",
                        help='Skip VAE training and reuse saved stage1 .pth and train_z.npy (stage2만 재학습)')
    if debug:
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()
#%%
def main():
    #%%
    # 인터랙티브(#%% 셀) 실행이면 기본값, 스크립트 실행이면 CLI 인자 사용
    config = vars(get_args(debug=hasattr(sys, 'ps1') or 'ipykernel' in sys.modules)) # default configuration
    set_random_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Current device is', device)
    wandb.config.update(config)
    #%%
    dataset_module = importlib.import_module(f"datasets.preprocess")
    importlib.reload(dataset_module)

    CustomDataset = dataset_module.CustomDataset
    train_dataset = CustomDataset(
        config, train=True)
    train_dataloader = DataLoader(
        train_dataset, batch_size=config['batch_size1'], shuffle=True)
    #%%    
    # Stage 1 (VAE) 
    model_module = importlib.import_module('modules.model1')
    importlib.reload(model_module)
#%%
    config["d_numerical"] = train_dataset.num_continuous_features
    config['categories'] = train_dataset.num_categories # the number of unique value
#%%
    model1 = model_module.Model_VAE(
        num_layers = config['num_layers'],  
        d_numerical = config["d_numerical"], 
        categories = config['categories'], 
        d_token = config['d_token'],
        var = config['var'],
        factor=config["factor"],
        n_head = config["n_head"],
        bias = config['bias']).to(device) 
    model1.train()
    optimizer1 = torch.optim.Adam(
        model1.parameters(), 
        lr=config['lr1'],
        weight_decay=config["weight_decay1"])
    scheduler1 = ReduceLROnPlateau(optimizer1, mode='min', factor=0.95, patience=10)
    
    # Model Training
    s1_base = f"{config['dataset']}_{config['lr1']}_{config['d_token']}_{config['denoising_dim']}"
    s1_base += f"_{config['batch_size1']}_{config['batch_size2']}_{config['max_beta']}"
    s1_base += f"_{config['var']}_{config['lambda']}"
    s1_pth = f"./assets/models/{s1_base}/stage1_{s1_base}_{config['seed']}.pth"
    s1_z = f"./assets/models/{s1_base}/{s1_base}_{config['seed']}_train_z.npy"
    if config['reuse_stage1']:
        assert os.path.exists(s1_pth) and os.path.exists(s1_z), \
            f"--reuse_stage1: 저장된 stage1 결과가 없습니다 ({s1_pth}, {s1_z})"
        print(f"[stage1] 학습 생략, 재사용: {s1_pth}")
        model1.load_state_dict(torch.load(s1_pth, map_location=device))
        train_z = np.load(s1_z)
    else:
        train_module1 = importlib.import_module('modules.train1')
        importlib.reload(train_module1)
        train_z = train_module1.train_function(
            model1,
            train_dataset,
            train_dataloader,
            config,
            optimizer1,
            scheduler1,
            device
        )
    #%%
    # LLM 조건 임베딩: 각 학습 행의 라벨 → 템플릿 문장 → LLM 임베딩 [N_train, llm_dim]
    llm_module = importlib.import_module('modules.llm_embed')
    importlib.reload(llm_module)
    emb_base = f"{config['dataset']}_{config['lr1']}_{config['d_token']}_{config['denoising_dim']}"
    emb_base += f"_{config['batch_size1']}_{config['batch_size2']}_{config['max_beta']}"
    emb_base += f"_{config['var']}_{config['lambda']}"
    emb_cache = f"./assets/models/{emb_base}/{emb_base}_{config['seed']}_llm_emb.npy"
    embedder = llm_module.LLMEmbedder(config['llm_model'], device=device)
    llm_emb = llm_module.build_train_embeddings(config, train_dataset, embedder, cache_path=emb_cache)
    embedder.release()
    del embedder
    llm_emb = torch.tensor(llm_emb).float()
    config['llm_dim'] = llm_emb.shape[1]
    config['llm_centered'] = True  # 학습 평균 μ로 중심화된 임베딩 (추론도 같은 μ 사용)
    print('llm_emb is formulated by size:', llm_emb.shape)
    #%%
    train_z = torch.tensor(train_z).float()
    print("<train_z.shape> : ", train_z.shape)

    train_z = train_z[:, 1:, :]
    B, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim
    train_z = train_z.view(B, in_dim)
    print('train_z is formulated by size:',train_z.shape)
    
    # Stage 2 (Diffusion) 
    denoise_fn_module = importlib.import_module(f"modules.model2")
    importlib.reload(denoise_fn_module)
    denoise_fn = denoise_fn_module.MLPDiffusion(
        in_dim,
        config['denoising_dim'], ## fixed 1024
        llm_dim=config['llm_dim'],  # LLM 임베딩 차원 → llm_proj로 denoising_dim에 맞춤
    ).to(device)
    denoise_fn.train()
    
    model2 = denoise_fn_module.Model(
        denoise_fn = denoise_fn, hid_dim = in_dim
    ).to(device)
    
    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params_1 = count_parameters(model1) 
    num_params_2 = count_parameters(model2)
    print(f"Number of VAE Parameters: {num_params_1 / 1_000_000:.6f}M")
    print(f"Number of Diffusion Parameters: {num_params_2 / 1_000_000:.6f}M")
    wandb.log({"Number of VAE Parameters": num_params_1 / 1000000})
    wandb.log({"Number of Diffusion Parameters": num_params_2 / 1000000})

    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3, weight_decay=0)  
    scheduler2 = ReduceLROnPlateau(optimizer2, mode='min', factor=0.9, patience=20)
    model2.train()

    # Model Training
    train_module2 = importlib.import_module('modules.train2')
    importlib.reload(train_module2)
    print("TRAIN model2")
    train_module2.train_function(
        config,
        model2,
        train_z,
        llm_emb,
        optimizer2,
        scheduler2,
        device,
    )

    # Saving
    base_name = f"{config['dataset']}_{config['lr1']}_{config['d_token']}_{config['denoising_dim']}"
    base_name += f"_{config['batch_size1']}_{config['batch_size2']}_{config['max_beta']}"
    base_name += f"_{config['var']}_{config['lambda']}"
    model_dir = f"./assets/models/{base_name}/"
    if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    model_name = f"{base_name}_{config['seed']}"

    artifact = wandb.Artifact(
        "_".join(model_name.split("_")),  
        type='model',
        metadata=config
    ) 
    
    # artifact.add_file(f"./{model_dir}/stage1_{model_name}.pth")
    # artifact.add_file(f"./{model_dir}/stage1_{model_name}.pth")
    
    artifact.add_file('./main.py')
    artifact.add_file('./datasets/preprocess.py')
    artifact.add_file('./modules/train1.py')
    artifact.add_file('./modules/model1.py')
    artifact.add_file('./modules/train2.py')
    artifact.add_file('./modules/model2.py')

    wandb.log_artifact(artifact)
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
    
    
# %%
