#%%
import os
# LLM 토크나이저 사용 후 DataLoader worker를 fork할 때 나오는 경고/교착 방지
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import pandas as pd
import torch
import numpy as np
import argparse
import importlib
from datasets.preprocess import build_num_inverse_fn, build_cat_inverse_fn
from modules.utils import set_random_seed, noise_sample, recover_data, split_num_cat_target
from evaluation.clipped_coverage import ClippedDensityCoverage 
from prdc import compute_prdc
from dython.nominal import associations
from synthetic_eval import evaluation

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
    tags=["inference"], # put tags of this python project
)

def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    ### defualt
    parser.add_argument('--ver', type=int, default=0, 
                        help='ver for repeatable results')
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
    parser.add_argument('--epochs', default=4000, type=int,
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
    parser.add_argument("--latent_noise_variance", default=0.0, type=float,                    
                        help="The variance of initial sample")
    
    ### Inference factor
    ### LLM 조건 (자연어 프롬프트)
    parser.add_argument("--prompt", type=str,
                        default="A frog call recording of genus Hypsiboas.",
                        help='Natural-language description of the data to generate')
    parser.add_argument("--num_samples", type=int, default=None,
                        help='Number of synthetic samples (default: size of train set)')
    parser.add_argument("--artifact_alias", type=str, default="latest",
                        help='wandb artifact alias to load (e.g. latest, v0)')
    parser.add_argument("--coverage_k", default=5, type=int,
                        help="The nearest neighbor in inference procedure")
    
    if debug:
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()
    
#%%
def main():
    #%%
    # 인터랙티브(#%% 셀) 실행이면 기본값, 스크립트 실행이면 CLI 인자 사용
    config = vars(get_args(debug=hasattr(sys, 'ps1') or 'ipykernel' in sys.modules))
    model_name = f"{config['dataset']}_{config['lr1']}_{config['d_token']}_{config['denoising_dim']}"
    model_name += f"_{config['batch_size1']}_{config['batch_size2']}_{config['max_beta']}"
    model_name += f"_{config['var']}_{config['lambda']}"

    artifact = wandb.use_artifact(
        f"{project}/{model_name}_{config['ver']}:{config['artifact_alias']}",
        type='model')
    # 추론 전용 인자는 학습 metadata로 덮어쓰지 않음
    infer_only = {k: config[k] for k in ['prompt', 'num_samples', 'artifact_alias', 'ver', 'coverage_k']}
    for key, item in artifact.metadata.items():
        config[key] = item
    config.update(infer_only)
    assert 'llm_dim' in config, "이 artifact는 LLM 조건 없이 학습된 모델입니다. main.py로 다시 학습하세요."
    
    info = artifact.metadata.get("info", {})
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    set_random_seed(config["seed"])
    wandb.config.update(config)

    dataset_module = importlib.import_module(f"datasets.preprocess")
    importlib.reload(dataset_module)

    # Dataset
    CustomDataset = dataset_module.CustomDataset
    train_dataset = CustomDataset(
        config, train=True)
    test_dataset = CustomDataset(
        config, train='test', cont_scalers=train_dataset.cont_scalers, cat_scalers=train_dataset.cat_scalers )        
    
    num_inverse = build_num_inverse_fn(train_dataset.cont_scalers)
    cat_inverse = build_cat_inverse_fn(train_dataset.cat_scalers)
    
    train_z = np.load(f'./assets/models/{model_name}/{model_name}_{config["seed"]}_train_z.npy')
    train_z = torch.tensor(train_z).float()
    train_z = train_z[:, 1:, :]
    B, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim
    train_z = train_z.view(B, in_dim)
    config["token_dim"] = token_dim

    # Load Model Weights
    denoise_fn_module = importlib.import_module(f"modules.model2")
    importlib.reload(denoise_fn_module)

    # LLM 임베딩 차원: 학습 때 artifact metadata에 저장한 값 사용
    llm_dim = config['llm_dim']

# 기존 denoise_fn
    # denoise_fn = denoise_fn_module.MLPDiffusion(
    #     in_dim,
    #     config['denoising_dim'],  
    # ).to(device)
# 새 denoise_fn
    denoise_fn = denoise_fn_module.MLPDiffusion(
        in_dim,
        config['denoising_dim'],
        llm_dim=llm_dim,  # 추가된 부분
    ).to(device)
    model2 = denoise_fn_module.Model(
        denoise_fn = denoise_fn, hid_dim = in_dim
    ).to(device) 
    model2.load_state_dict(
        torch.load(
            f"./assets/models/{model_name}/stage2_{model_name}_{config['seed']}.pth",
            map_location=device,
        )
    )
    info['model_dir'] = f"./assets/models/{model_name}/stage1_{model_name}_{config['seed']}.pth"
    model2.eval()

    # Synthetic Data Generation
    num_samples = config['num_samples'] or B

    # 자연어 프롬프트 → 조건 임베딩 [num_samples, llm_dim]
    from modules.llm_embed import LLMEmbedder, load_cond_meta, parse_prompt
    print(f"[LLM] prompt: {config['prompt']}")
    embedder = LLMEmbedder(config['llm_model'], device=device)
    emb_cache = f"./assets/models/{model_name}/{model_name}_{config['seed']}_llm_emb.npy"
    cond = load_cond_meta(emb_cache) if config.get('llm_centered') else None
    constraints = {}
    if cond is not None:
        # 프롬프트 → 라벨 제약 → 제약에 맞는 학습 조합을 샘플마다 빈도 비례로 선택 (중심화된 임베딩)
        constraints, _, prompt_emb = parse_prompt(
            config['prompt'], cond, embedder, config['dataset'], num_samples, seed=config['seed'])
    else:
        # 구버전 모델(중심화 이전): 프롬프트 임베딩을 그대로 반복
        prompt_emb = embedder.encode([config['prompt']]).repeat(num_samples, 1)
    embedder.release()
    del embedder
    assert prompt_emb.shape[1] == llm_dim, f"임베딩 차원 불일치: {prompt_emb.shape[1]} != {llm_dim}"
    llm_emb = prompt_emb.to(device)
    wandb.log({"prompt": config['prompt']})
    wandb.config.update({"constraints": constraints}, allow_val_change=True)
    import re
    slug = re.sub(r"[^0-9A-Za-z가-힣]+", "_", config['prompt']).strip("_")[:40] or "prompt"

# 기존 x_next
    # x_next = noise_sample(model2.denoise_fn_D, num_samples, in_dim, config['latent_noise_variance'])
# 새 x_next
    x_next = noise_sample(
        model2.denoise_fn_D, 
        num_samples, 
        in_dim, 
        config['latent_noise_variance'],
        device=device,
        llm_emb=llm_emb  # 추가된 부분
    )
    x_next = x_next * 2 + train_z.mean(0).to(device)
    
    syn_data = x_next.float().cpu().numpy()
    syn_data = syn_data.astype(np.float32)
    #%%
    os.makedirs("./syn_data", exist_ok=True)
    df_synth = pd.DataFrame(syn_data, columns=[f"dim_{i}" for i in range(syn_data.shape[1])])
    output_path = f"./syn_data/synthetic_prior_{model_name}_{config['seed']}_{slug}.csv"
    df_synth.to_csv(output_path, index=False)
    #%%
    syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, cat_inverse, config, device) 
    
    syn_df = recover_data(syn_num, syn_cat, syn_target, info)
    
    idx_name_mapping = info['idx_name_mapping']
    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}
    
    syn_df.rename(columns = idx_name_mapping, inplace=True)
    syn_df[train_dataset.categorical_features] = syn_df[train_dataset.categorical_features].astype(int)
    syn_df[train_dataset.integer_features] = syn_df[train_dataset.integer_features].astype(int)
    syn_df[train_dataset.continuous_features] = syn_df[train_dataset.continuous_features].astype(np.float32)
    #%%
    # 프롬프트 조건 확인: 생성 데이터의 범주 분포 (코드 → 원래 이름으로 복원해서 출력)
    from datasets.raw_data import load_raw_data
    raw_full = load_raw_data(config)[0]
    print(f"\n=== Generated category distribution | prompt: {config['prompt']} ===")
    for col in train_dataset.categorical_features:
        cats = raw_full[col].astype("category").cat.categories
        names = syn_df[col].clip(0, len(cats) - 1).map(lambda c: cats[c])
        print(f"[{col}]\n{names.value_counts(normalize=True).round(3).to_string()}\n")

    # 조건부 지표: 조합 일관성 / 제약 만족률 / 조건에 맞는 실제 데이터 대비 coverage
    cat_cols = train_dataset.categorical_features
    name_of = {col: raw_full[col].astype("category").cat.categories for col in cat_cols}
    code_of = {col: {n: i for i, n in enumerate(name_of[col])} for col in cat_cols}
    real_tr = train_dataset.raw_data
    valid = set(map(tuple, real_tr[cat_cols].astype(int).drop_duplicates().to_numpy()))
    syn_codes = syn_df[cat_cols].astype(int).to_numpy()
    consistency = float(np.mean([tuple(r) in valid for r in syn_codes]))
    sat_mask = np.ones(len(syn_df), dtype=bool)
    real_mask = np.ones(len(real_tr), dtype=bool)
    for col, names in constraints.items():
        codes = [code_of[col][n] for n in names]
        sat_mask &= syn_df[col].astype(int).isin(codes).to_numpy()
        real_mask &= real_tr[col].astype(int).isin(codes).to_numpy()
    satisfaction = float(sat_mask.mean())
    print(f"조합 일관성 비율 (실제 존재하는 (F,G,S) 조합): {consistency:.3f}")
    print(f"제약 만족률 {constraints}: {satisfaction:.3f}")
    wandb.log({"cond/consistency": consistency, "cond/satisfaction": satisfaction})
    if constraints:
        real_cond = real_tr[real_mask].astype('float32')
        print(f"조건에 맞는 실제 train 행: {len(real_cond)}")
        k = min(config['coverage_k'], max(len(real_cond) - 1, 1))
        cov_c = compute_prdc(real_features=real_cond, fake_features=syn_df.astype('float32'),
                             nearest_k=k)
        print(f"Coverage (조건 일치 train 기준, k={k}): {cov_c['coverage']:.4f}, "
              f"Precision: {cov_c['precision']:.4f}, Recall: {cov_c['recall']:.4f}")
        wandb.log({"cond/coverage": cov_c['coverage'], "cond/precision": cov_c['precision'],
                   "cond/recall": cov_c['recall']})
    print("\n※ 아래 지표들은 '전체' 실제 데이터 기준입니다 (조건부 생성이면 낮게 나오는 게 정상).")
    #%%
    # Evaluation
    """ Synthetic Eval packages """
    results = evaluation.evaluate(
        syn_df,train_dataset.raw_data.astype('float32'), test_dataset.raw_data.astype('float32'), 
        train_dataset.ClfTarget, train_dataset.continuous_features, train_dataset.categorical_features, device
        )
    '''print results'''
    for x, y in results._asdict().items():
        print(f"{x}: {y:.3f}")
        wandb.log({f"{x}": y})
    #%%
    print("Coverage for Diversity Evaluation ...")
    coverage_k = config['coverage_k']
    coverage = compute_prdc(real_features=train_dataset.raw_data.astype('float32'),
                            fake_features=syn_df,
                            nearest_k=coverage_k)
    print("Coverage : ",coverage["coverage"])
    wandb.log({"Coverage":coverage["coverage"]})
    
    print("Clipped Coverage(Train Set) for Diversity Evaluation ...")
    CDC = ClippedDensityCoverage(train_dataset.raw_data.to_numpy(dtype='float32'),
                                K=coverage_k,
                                n_jobs=8,)
    
    print("Clipped_Coverage(Train Set) : ", CDC.ClippedCoverage(syn_df.to_numpy(dtype='float32')))
    wandb.log({"Clipped_Coverage(Train Set)" : CDC.ClippedCoverage(syn_df.to_numpy(dtype='float32'))})
    #%%
    syn_df.to_csv(f"./syn_data/{model_name}_{config['seed']}_{config['latent_noise_variance']}_{slug}.csv")
    print("Pairwise correlation difference (PCD)...")

    syn_asso = associations(
        syn_df, nominal_columns=train_dataset.categorical_features,
        compute_only=True)
    true_asso = associations(
        train_dataset.raw_data, nominal_columns=train_dataset.categorical_features,
        compute_only=True)
    pcd_corr = np.linalg.norm(true_asso["corr"] - syn_asso["corr"])
    print("Pairwise correlation difference (PCD) : ",pcd_corr)

    wandb.log({"PCD":pcd_corr})
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%
    