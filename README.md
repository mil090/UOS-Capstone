# UOS-Capstone
2026. 2학기 통계자료분석캡스톤디자인 주입식 학습 팀 리포지토리

## 개요
원하는 데이터를 자연어로 설명하면, 그 조건에 맞는 표 데이터만 생성하는 것이 목표입니다.
[SPT](https://github.com/Optim-Lab/SPT)의 stage 2(diffusion denoiser)에 LLM 텍스트 임베딩을 조건으로 주입하는 방식입니다.

```
프롬프트 ─▶ 라벨 제약 해석 ─▶ LLM 임베딩(Qwen2.5-0.5B, 중심화) ─▶ diffusion 샘플링 ─▶ VAE 디코딩 ─▶ 표 데이터
```

- **학습**
  - 각 행의 라벨로 템플릿 문장을 만들어요. 예: `A frog call recording of species …, genus …, family ….`
  - 문장을 임베딩한 뒤 고유 문장들의 평균을 빼서 중심화해요. 그 임베딩을 `llm_proj`로 denoiser에 더해요.
- **추론**
  - 프롬프트에서 과·속·종 제약을 찾아요. 라벨 이름이나 한국어·영어 일반명은 규칙으로 찾고, 그 밖의 문장은 LLM이 해석해요.
  - 제약에 맞는 학습 조합을 빈도에 비례해 샘플마다 골라 조건으로 넣어요.

## 결과 (anuran, 2026-11-11)
stage 1 1000 에폭, stage 2 3000 에폭(early stopping)으로 학습했어요. 프롬프트마다 5,756행을 생성했어요.

| 프롬프트 | 해석된 조건 | 조합 일관성 | 제약 만족률 | 조건부 Coverage | Precision |
| --- | --- | --- | --- | --- | --- |
| A frog call recording of genus Hypsiboas. | Genus = Hypsiboas | 1.000 | 1.000 | 1.000 | 0.984 |
| A frog call recording of genus Adenomera. | Genus = Adenomera | 1.000 | 1.000 | 1.000 | 0.997 |
| 청개구리과 개구리 울음소리 데이터를 생성해 줘 | Family = Hylidae | 1.000 | 1.000 | 0.999 | 0.984 |

- 조건부 Coverage와 Precision은 조건에 맞는 실제 train 데이터를 기준으로 계산했어요.
- 생성된 종 비율은 학습 데이터 비율과 거의 같아요. 예: Hypsiboas 프롬프트에서 Cordobae 70.9%, Cinerascens 29.1%

## 실행
```bash
cd SPT && mkdir -p logs
sbatch run_full.sh   # 학습 + 프롬프트 3개 추론 (A10 1장, 약 27분)
```
직접 실행하려면 아래처럼 해요.
```bash
python main.py $COMMON --seed 0 --epochs 1000 --epochs_2 3000     # --reuse_stage1: 저장된 VAE 재사용
python inference.py $COMMON --ver 0 --prompt "청개구리과 개구리 울음소리 데이터를 생성해 줘"
python tools/check_recon.py $COMMON --seed 0                       # VAE 복원 정확도 진단
```
`$COMMON`은 `run_full.sh`에 정의된 하이퍼파라미터예요.

### 결과 확인 위치
- 범주 분포와 조건부 지표: `logs/`
- 지표: wandb의 `cond/*`
- 생성 데이터: `syn_data/`

### 주요 파일
- `SPT/modules/llm_embed.py`
- `SPT/modules/model2.py`
- `SPT/main.py`
- `SPT/inference.py`

## 한계와 다음 단계
- 조건은 학습 데이터에 있는 라벨 조합으로만 표현돼요. 그래서 MFCC 같은 수치 조건이나 학습에 없는 조합은 반영되지 않아요.
- 다음 단계
  - 암기 여부 확인(DCR)
  - 다른 데이터셋으로 확장
  - 여러 seed로 반복 실험
  - 더 큰 LLM으로 자유 문장 해석 비교

---

> 아래는 원본 SPT 레포지토리([Optim-Lab/SPT](https://github.com/Optim-Lab/SPT))의 README입니다.

# Generating High-Diversity Synthetic Tabular Data via a Less-Constrained Prior

This is the official implementation of the paper  
**"Generating High-Diversity Synthetic Tabular Data via a Less-Constrained Prior"** (IJCAI-ECAI 2026).

---

## 📦 Environment Setup

We recommend using a virtual environment.

Install the required dependencies using:

```bash
pip install -r requirements.txt
````

---

## 🔑 Weights & Biases (wandb) Setup

This codebase uses **Weights & Biases (wandb)** for experiment tracking.

1. Create a wandb account at [https://wandb.ai](https://wandb.ai)
2. Log in from the terminal:

```bash
wandb login
```

3. In `main.py` or `inference.py`, set your wandb information at the top of the file:

```python
project = "YOUR_PROJECT_NAME"
entity  = "YOUR_WANDB_USERNAME"
```

---

## 🏋️‍♂️ Training / Inference

This repository provides implementations of our proposed method (**SPT**) as well as baseline methods such as **CTGAN**, **TVAE**, **TabMT**, **TabDDPM**, **STaSy**, **TabSyn**, and **AutoDiff**.

The overall workflow is as follows:

1. Prepare a wandb account and configure it in the code
2. Move to the directory of the desired model (e.g., `SPT`, `TabSyn`)
3. Train the model using `main.py`
4. Generate synthetic samples using `inference.py`

---

### Step 1: Move to the Model Directory

First, navigate to the directory corresponding to the baseline or method you wish to execute. For example:

```bash
cd SPT
# or
cd baseline/TabSyn
# or
cd baseline/TabDDPM
```

---

### Step 2: Training

To train a model, run the following command:

```bash
python main.py --dataset [NAME_OF_DATASET] [ADDITIONAL_ARGS]
```

Here, `[ADDITIONAL_ARGS]` corresponds to model-specific hyperparameters, which are summarized below.

#### Required arguments by model

| Model     | Required Arguments                                                                                                     |
| ------------ | ---------------------------------------------------------------------------------------------------------------------- |
| **SPT**      | `dataset`, `var`, `d_token`, `denoising_dim`, `lr1`, `batch_size1`, `batch_size2`, `max_beta`                          |
| **TabSyn**   | `dataset`, `d_token`, `denoising_dim`, `lr1`, `batch_size1`, `batch_size2`, `max_beta`                                 |
| **TabDDPM**  | `dataset`, `weight_decay`, `lr`, `dim_embed`, `batch_size`, `num_layers`                                               |
| **CTGAN**    | `dataset`, `latent_dim`, `batch_size`, `epochs`, `generator_dim`, `discriminator_dim`                                  |
| **TVAE**     | `dataset`, `latent_dim`, `batch_size`, `epochs`, `loss_factor`                                                         |
| **TabMT**    | `dataset`, `dim_transformer`, `num_transformer_heads`, `num_transformer_layer`, `batch_size`, `epochs`, `max_clusters` |
| **STaSy**    | `dataset`, `model.sigma_min`, `model.sigma_max`, `optim.lr`, `model.beta0`, `model.alpha0`                             |
| **AutoDiff** | `dataset`, `d_token`, `denoising_dim`, `lr1`, `batch_size1`, `batch_size2`                                             |


#### Notes

1. Supported datasets for `--dataset` are:

```
abalone, adult, anuran, banknote, breast, concrete,
kings, letter, loan, redwine, shoppers, whitewine
```

2. To ensure reproducibility, you can fix the random seed during training by adding the `--seed` argument.

3. Below is an example command for training **SPT** on the **redwine** dataset with a fixed seed:

```bash
python main.py --dataset redwine --var 0.1 --d_token 4 --denoising_dim 1024 \
               --lr1 0.001 --batch_size1 512 --batch_size2 512 --max_beta 0.01 --seed 0
```

---
### Step 3: Inference (Synthetic Data Generation)

After training, synthetic samples can be generated using the following command:

```bash
python inference.py --dataset [NAME_OF_DATASET] [ADDITIONAL_ARGS]
```

Here, `[ADDITIONAL_ARGS]` must be **identical to those used during training**, as they are required to correctly load the trained model checkpoint.

#### Notes

1. The same set of hyperparameters used in the training phase must be provided during inference.

2. During inference, the argument `--ver` is used to specify the experiment version stored in wandb.
   * `--ver` controls which trained checkpoint is loaded and does not need to match `seed`

3. Below is an example command for generating synthetic data using **SPT** trained on the **redwine** dataset:

```bash
python inference.py --dataset redwine --var 0.1 --d_token 4 --denoising_dim 1024 \
                     --lr1 0.001 --batch_size1 512 --batch_size2 512 --max_beta 0.01 --ver 0
```

   This command loads the trained SPT model corresponding to version `ver = 0` from wandb and generates synthetic samples for the `redwine` dataset.

---

## ⚙ Example & Reproducibility

### Quick Start with Jupyter Notebook

We provide a comprehensive example to ensure easy reproducibility of our results.

* **File Location:** `SPT/example.ipynb`
* **Content:** This notebook contains the entire pipeline for the **anuran dataset**, including:
1. **Training:** Step-by-step model training using the SPT method.
2. **Inference:** Generating synthetic tabular data from the trained model.
3. **Evaluation:** Measuring the quality of synthetic data using all metrics (GoF, MMD, PCD, Coverage, etc.).

You can run this notebook to verify the performance and understand the workflow of our proposed method.

---

## Cite
```
@inproceedings{ijcai2026p525,
  title     = {Generating High-Diversity Synthetic Tabular Data via Less-Constrained Prior},
  author    = {Park, Sanghun and Lim, Jaesung and Jeon, Jong-June and An, Seunghwan},
  booktitle = {Proceedings of the Thirty-Fifth International Joint Conference on
               Artificial Intelligence, {IJCAI-26}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Diego Calvanese},
  pages     = {4714--4722},
  year      = {2026},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2026/525},
  url       = {https://doi.org/10.24963/ijcai.2026/525},
}
```

## Acknowledgement
This research was supported by the National Research Foundation of Korea(NRF) grant funded by the Korea government(MSIT) (NO. RS-2022-NR068754)
