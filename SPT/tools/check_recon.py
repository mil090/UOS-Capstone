"""
VAE(stage1) 복원 정확도 진단.
저장된 train_z를 inference.py와 '같은 디코딩 경로'(split_num_cat_target → recover_data → 이름 복원)로
되돌려서 원본 학습 데이터와 비교한다. wandb 불필요.

- 범주형 정확도가 높으면(>95%): VAE·디코딩 정상 → 조합 불일치는 stage2/조건 문제
- 낮으면: VAE 학습 부족 → stage1 에폭 증가 필요

실행 (SPT/SPT에서, 학습 때와 같은 인자):
  python tools/check_recon.py --dataset anuran --var 0.1 --d_token 4 --denoising_dim 1024 \
      --lr1 0.001 --batch_size1 512 --batch_size2 512 --max_beta 0.01 --seed 0
"""
import os
import sys
import argparse

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(HERE)
sys.path.insert(0, HERE)

import numpy as np
import torch

from datasets.preprocess import CustomDataset, build_num_inverse_fn, build_cat_inverse_fn
from datasets.raw_data import load_raw_data
from modules.utils import split_num_cat_target, recover_data


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='anuran')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--test_size', type=float, default=0.2)
    p.add_argument('--batch_size1', type=int, default=512)
    p.add_argument('--batch_size2', type=int, default=512)
    p.add_argument('--var', type=float, default=0.1)
    p.add_argument('--lr1', type=float, default=0.001)
    p.add_argument('--d_token', type=int, default=4)
    p.add_argument('--num_layers', type=int, default=2)
    p.add_argument('--max_beta', type=float, default=1e-2)
    p.add_argument('--lambda', type=float, default=0.7)
    p.add_argument('--factor', type=int, default=32)
    p.add_argument('--n_head', type=int, default=1)
    p.add_argument('--bias', type=bool, default=True)
    p.add_argument('--denoising_dim', type=int, default=1024)
    return vars(p.parse_args())


def main():
    config = get_args()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    base = f"{config['dataset']}_{config['lr1']}_{config['d_token']}_{config['denoising_dim']}"
    base += f"_{config['batch_size1']}_{config['batch_size2']}_{config['max_beta']}"
    base += f"_{config['var']}_{config['lambda']}"
    model_dir = f"./assets/models/{base}"

    train_dataset = CustomDataset(config, train=True)
    config['d_numerical'] = train_dataset.num_continuous_features
    config['categories'] = train_dataset.num_categories
    info = dict(train_dataset.info)
    info['model_dir'] = f"{model_dir}/stage1_{base}_{config['seed']}.pth"

    train_z = np.load(f"{model_dir}/{base}_{config['seed']}_train_z.npy")
    z = train_z[:, 1:, :].reshape(train_z.shape[0], -1).astype(np.float32)
    print(f"device={device}, train_z {train_z.shape} -> {z.shape}")

    num_inverse = build_num_inverse_fn(train_dataset.cont_scalers)
    cat_inverse = build_cat_inverse_fn(train_dataset.cat_scalers)
    with torch.no_grad():
        syn_num, syn_cat, syn_target = split_num_cat_target(z, info, num_inverse, cat_inverse, config, device)
    rec = recover_data(syn_num, syn_cat, syn_target, info)
    rec.rename(columns={int(k): v for k, v in info['idx_name_mapping'].items()}, inplace=True)

    real = train_dataset.raw_data
    cats = train_dataset.categorical_features
    conts = train_dataset.continuous_features

    print("\n=== 범주형 복원 정확도 (train_z → decode vs 원본) ===")
    for col in cats:
        acc = (rec[col].astype(int).to_numpy() == real[col].astype(int).to_numpy()).mean()
        print(f"  {col:8s}: {acc:.4f}")
    all_acc = (rec[cats].astype(int).to_numpy() == real[cats].astype(int).to_numpy()).all(1).mean()
    print(f"  (3개 모두 일치): {all_acc:.4f}")

    valid = set(map(tuple, real[cats].astype(int).drop_duplicates().to_numpy()))
    consist = np.mean([tuple(r) in valid for r in rec[cats].astype(int).to_numpy()])
    print(f"\n조합 일관성 비율 (복원 행의 (F,G,S)가 실제 존재 조합인 비율): {consist:.4f}")

    mae = np.abs(rec[conts].astype(float).to_numpy() - real[conts].astype(float).to_numpy()).mean()
    std = real[conts].astype(float).to_numpy().std(0).mean()
    print(f"MFCC 복원 MAE: {mae:.4f}  (참고: 특성 평균 표준편차 {std:.4f})")

    # 이름으로 본 혼동 상위 (틀린 경우)
    raw_full = load_raw_data(config)[0]
    sp_names = raw_full['Species'].astype('category').cat.categories if 'Species' in cats else None
    if sp_names is not None:
        wrong = rec['Species'].astype(int).to_numpy() != real['Species'].astype(int).to_numpy()
        if wrong.any():
            import pandas as pd
            pairs = pd.Series([f"{sp_names[a]} -> {sp_names[b]}" for a, b in zip(
                real['Species'].astype(int).to_numpy()[wrong], rec['Species'].astype(int).to_numpy()[wrong])])
            print("\nSpecies 오복원 상위:\n" + pairs.value_counts().head(8).to_string())


if __name__ == '__main__':
    main()
