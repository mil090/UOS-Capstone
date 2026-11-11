"""
LLM 기반 조건 임베딩 생성 모듈.

자연어 문장을 decoder LLM(기본: Qwen2.5-0.5B-Instruct)에 통과시키고,
마지막 hidden state를 attention mask 기준으로 mean-pooling한 뒤 L2 정규화하여
고정 길이 임베딩 벡터([N, hidden])를 만든다.

- 학습: 각 데이터 행의 범주형 라벨로 템플릿 문장을 만들고 임베딩
        → 고유 문장 임베딩의 평균 μ를 빼고 다시 L2 정규화(중심화) → diffusion 조건
- 추론: 사용자 프롬프트에서 라벨 제약(과/속/종)을 해석(parse_prompt)
        → 제약에 맞는 학습 조합을 샘플마다 빈도 비례로 뽑아 그 조합의 (중심화된) 임베딩을 조건으로 주입
        → 해석 실패 시 프롬프트 자체를 임베딩해 같은 μ로 중심화
"""
import json
import os
import re

import numpy as np
import pandas as pd
import torch

DEFAULT_LLM = "Qwen/Qwen2.5-0.5B-Instruct"


class LLMEmbedder:
    def __init__(self, model_name=DEFAULT_LLM, device=None, max_length=128):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device(device) if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # CausalLM으로 로드: .model(=AutoModel과 동일한 본체)로 임베딩, generate()로 프롬프트 해석
        self.lm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(self.device)
        self.lm.eval()
        self.model = self.lm.model
        self.max_length = max_length
        self.dim = self.lm.config.hidden_size

    @torch.no_grad()
    def encode(self, texts, batch_size=32):
        """texts(list[str]) -> torch.FloatTensor [N, hidden] (CPU, L2 정규화)"""
        if isinstance(texts, str):
            texts = [texts]
        outs = []
        self.tokenizer.padding_side = "right"
        for i in range(0, len(texts), batch_size):
            batch = self.tokenizer(
                texts[i:i + batch_size], padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt").to(self.device)
            hidden = self.model(**batch).last_hidden_state.float()      # [b, L, H]
            mask = batch["attention_mask"].unsqueeze(-1).float()        # [b, L, 1]
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            outs.append(torch.nn.functional.normalize(pooled, dim=-1).cpu())
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def chat(self, system, user, max_new_tokens=96):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = self.tokenizer(text, return_tensors="pt").to(self.device)
        out = self.lm.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                               pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)

    def release(self):
        """GPU 메모리 해제 (diffusion 학습 전에 호출)"""
        del self.model
        del self.lm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 데이터 행 -> 텍스트
# ---------------------------------------------------------------------------
def row_to_text(row, dataset):
    """범주형 라벨(원래 이름)이 담긴 dict/Series 한 행을 설명 문장으로 변환"""
    if dataset == "anuran":
        return (f"A frog call recording of species {row['Species']}, "
                f"genus {row['Genus']}, family {row['Family']}.")
    # 그 외 데이터셋: 범주형 컬럼들을 나열하는 일반 템플릿
    parts = ", ".join(f"{k} is {v}" for k, v in row.items())
    return f"A {dataset} data record where {parts}."


def decode_categories(config, train_dataset):
    """
    preprocess.CustomDataset.raw_data의 범주형 컬럼은 정수 코드로 바뀌어 있으므로,
    원본 데이터에서 같은 방식(astype('category'))으로 카테고리 순서를 구해 이름으로 복원한다.
    """
    from datasets.raw_data import load_raw_data

    raw, _, cat_features, _, _ = load_raw_data(config)
    cat_df = pd.DataFrame(index=train_dataset.raw_data.index)
    for col in cat_features:
        categories = raw[col].astype("category").cat.categories
        cat_df[col] = categories[train_dataset.raw_data[col].astype(int).to_numpy()]
    return cat_df


def center(emb, mean):
    """임베딩에서 학습 평균 μ를 빼고 다시 L2 정규화 (torch/numpy 모두 허용)"""
    emb = torch.as_tensor(emb).float() - torch.as_tensor(mean).float()
    return torch.nn.functional.normalize(emb, dim=-1)


def cond_paths(cache_path):
    """학습 때 저장하고 추론 때 읽는 조건 관련 파일 경로들"""
    stem = cache_path[:-len(".npy")]
    return {"mean": f"{stem}_mean.npy", "uniq_emb": f"{stem}_uniq_emb.npy",
            "uniq_meta": f"{stem}_uniq.json"}


def build_train_embeddings(config, train_dataset, embedder, cache_path=None):
    """
    train_dataset의 각 행(train_z와 같은 순서)에 대한 '중심화된' 텍스트 임베딩 [N_train, hidden] 생성.
    고유한 문장만 LLM에 통과시킨 뒤 평균 μ로 중심화하고 행에 다시 매핑한다.
    cache_path가 있으면 행별 임베딩(.npy)과 μ, 고유 문장 임베딩/라벨/빈도를 저장한다.
    """
    cat_df = decode_categories(config, train_dataset)
    texts = [row_to_text(r, config["dataset"]) for r in cat_df.to_dict("records")]
    uniq, first_idx, inverse, counts = np.unique(
        np.array(texts), return_index=True, return_inverse=True, return_counts=True)
    print(f"[LLM] {len(texts)} rows -> {len(uniq)} unique texts. e.g. '{uniq[0]}'")
    raw_uniq = embedder.encode(list(uniq))                               # [U, H]
    mean = raw_uniq.mean(0)                                              # 문장 빈도와 무관한 평균
    uniq_emb = center(raw_uniq, mean)                                    # [U, H]
    cos = uniq_emb @ uniq_emb.T
    off = cos[~torch.eye(len(uniq), dtype=bool)]
    print(f"[LLM] centered unique-text cosine: mean {off.mean():.3f}, max {off.max():.3f} "
          f"(before: {(raw_uniq @ raw_uniq.T)[~torch.eye(len(uniq), dtype=bool)].mean():.3f})")
    emb = uniq_emb[torch.from_numpy(inverse)].numpy().astype(np.float32)  # [N, H]
    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, emb)
        with open(cache_path.replace(".npy", "_texts.txt"), "w") as f:
            f.write("\n".join(uniq.tolist()))
        p = cond_paths(cache_path)
        np.save(p["mean"], mean.numpy().astype(np.float32))
        np.save(p["uniq_emb"], uniq_emb.numpy().astype(np.float32))
        meta = [{"text": t, "count": int(c), "labels": cat_df.iloc[int(i)].to_dict()}
                for t, c, i in zip(uniq.tolist(), counts, first_idx)]
        with open(p["uniq_meta"], "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
    return emb


# ---------------------------------------------------------------------------
# 추론: 프롬프트 -> 라벨 제약 -> 조건 임베딩
# ---------------------------------------------------------------------------
# 한국어 별칭 (0.5B LLM이 한국어 분류명을 잘 모르므로 규칙으로 보완)
KO_ALIASES = {
    "anuran": {
        "Family": {"두꺼비": "Bufonidae", "독화살개구리": "Dendrobatidae",
                   "청개구리": "Hylidae", "남방개구리": "Leptodactylidae",
                   "휘파람개구리": "Leptodactylidae",
                   # 영어 일반명
                   "poison dart": "Dendrobatidae", "poison frog": "Dendrobatidae",
                   "toad": "Bufonidae", "tree frog": "Hylidae", "treefrog": "Hylidae"},
    }
}


def _norm(s):
    return re.sub(r"[^0-9a-z]", "", str(s).lower())


def _rule_match(prompt, vocab, dataset):
    """프롬프트 단어(및 인접 두 단어 결합)와 라벨 이름을 정확히 비교 → {col: [names]}"""
    words = [_norm(w) for w in re.findall(r"[A-Za-z0-9]+", prompt)]
    tokens = set(words) | {a + b for a, b in zip(words, words[1:])}
    found = {}
    for col, names in vocab.items():
        hit = [n for n in names if _norm(n) in tokens]
        for ko, name in KO_ALIASES.get(dataset, {}).get(col, {}).items():
            if ko in prompt.lower() and name not in hit:
                hit.append(name)
        if hit:
            found[col] = hit
    return found


def _llm_match(prompt, vocab, embedder):
    """Qwen에게 라벨 목록을 주고 JSON으로 제약을 뽑게 한 뒤 vocab으로 검증"""
    keys = list(vocab.keys())
    allowed = "\n".join(f"- {c}: {', '.join(vocab[c])}" for c in keys)
    system = ("You convert a user's request into label constraints. Answer with a single JSON object "
              f"with keys {keys}. Each value must be one of the allowed labels, copied exactly, "
              "or null. Use null unless the request clearly refers to that label. "
              "If the request does not name any specific group, all values are null. No explanation.")
    null_ex = json.dumps({k: None for k in keys})
    user = (f"Allowed labels:\n{allowed}\n\n"
            f"Request: generate some frog data\nJSON: {null_ex}\n\n"
            f"Request: {prompt}\nJSON:")
    out = embedder.chat(system, user)
    print(f"[LLM] parse output: {out.strip()}")
    m = re.search(r"\{.*\}", out, re.S)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    found = {}
    for col in keys:
        v = obj.get(col) or obj.get(col.lower())
        cand = [n for n in vocab[col] if v is not None and _norm(n) == _norm(v)]
        if cand:
            found[col] = cand
    return found


def load_cond_meta(cache_path):
    p = cond_paths(cache_path)
    if not os.path.exists(p["uniq_meta"]):
        return None
    with open(p["uniq_meta"]) as f:
        meta = json.load(f)
    return {"meta": meta, "mean": np.load(p["mean"]), "uniq_emb": np.load(p["uniq_emb"])}


def parse_prompt(prompt, cond, embedder, dataset, num_samples, seed=0):
    """
    프롬프트 → (제약 dict, 후보 조합 목록, 조건 임베딩 [num_samples, H]).
    cond: load_cond_meta() 결과 (학습 조합, μ, 중심화된 조합 임베딩)
    """
    meta = cond["meta"]
    cols = list(meta[0]["labels"].keys())
    vocab = {c: sorted({m["labels"][c] for m in meta}) for c in cols}

    constraints = _rule_match(prompt, vocab, dataset)
    source = "rule"
    if not constraints:
        constraints = _llm_match(prompt, vocab, embedder)
        source = "llm"

    cand = [i for i, m in enumerate(meta)
            if all(m["labels"][c] in names for c, names in constraints.items())]
    if constraints and cand:
        w = np.array([meta[i]["count"] for i in cand], dtype=np.float64)
        rng = np.random.default_rng(seed)
        pick = rng.choice(cand, size=num_samples, p=w / w.sum())
        emb = torch.as_tensor(cond["uniq_emb"])[torch.as_tensor(pick)]
        print(f"[LLM] constraints ({source}): {constraints}")
        print("[LLM] candidate combos (sampling weight):")
        for i, wi in zip(cand, w / w.sum()):
            print(f"   {wi:.3f}  {meta[i]['text']}")
        return constraints, [meta[i]["labels"] for i in cand], emb

    # 제약 없음/모순 → 학습 분포 전체(모든 조합, 빈도 비례)에서 샘플링
    print(f"[LLM][경고] 프롬프트에서 유효한 라벨 제약을 찾지 못했습니다 (constraints={constraints}, "
          f"candidates={len(cand)}). 학습 분포 전체 조합에서 샘플링합니다.")
    w = np.array([m["count"] for m in meta], dtype=np.float64)
    pick = np.random.default_rng(seed).choice(len(meta), size=num_samples, p=w / w.sum())
    emb = torch.as_tensor(cond["uniq_emb"])[torch.as_tensor(pick)]
    return {}, [m["labels"] for m in meta], emb
