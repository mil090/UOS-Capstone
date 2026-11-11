"""
LLM 임베딩 방식 비교 (mean pooling vs PromptEOL 마지막 토큰, 중심화 여부).
- train-train cos: 서로 다른 라벨 문장끼리의 유사도 (낮을수록 조건 구분이 잘 됨)
- query top3: 자연어 프롬프트(영/한)가 어떤 학습 문장과 가장 가까운지
실행: python tools/compare_embeddings.py  (SPT/SPT 디렉토리에서)
"""
import time
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

t0 = time.time()
dev = "cuda" if torch.cuda.is_available() else "cpu"
n = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(n)
m = AutoModel.from_pretrained(n).to(dev).eval()
print(f"device={dev}, model loaded in {time.time()-t0:.1f}s", flush=True)

raw = pd.read_csv('../data/Frogs_MFCCs.csv')
combos = raw[['Family', 'Genus', 'Species']].drop_duplicates().values
train = [f"A frog call recording of species {s}, genus {g}, family {f}." for f, g, s in combos]
names = [f"{g}/{s}" for f, g, s in combos]
queries = {
    "EN Hypsiboas": "Generate frog calls of the genus Hypsiboas",
    "KO Hypsiboas": "Hypsiboas 속 개구리 울음소리 데이터를 생성해 줘",
    "KO Adenomera": "Adenomera 속 개구리 데이터를 만들어 줘",
}


@torch.no_grad()
def emb(texts, mode):
    out = []
    for t in texts:
        if mode == "eol":
            t = f'This sentence : "{t}" means in one word:"'
        b = tok(t, return_tensors='pt').to(dev)
        h = m(**b).last_hidden_state[0].float()
        out.append(h.mean(0) if mode == "mean" else h[-1])
    return torch.stack(out).cpu()


for mode in ["mean", "eol"]:
    T = emb(train, mode)
    Q = emb(list(queries.values()), mode)
    for center in [False, True]:
        mu = T.mean(0) if center else 0
        Tn = F.normalize(T - mu, dim=-1)
        Qn = F.normalize(Q - mu, dim=-1)
        S = Tn @ Tn.T
        off = S[~torch.eye(len(S), dtype=bool)]
        print(f"\n## {mode} center={center}: train-train cos mean={off.mean():.3f} max={off.max():.3f}", flush=True)
        for k, row in zip(queries, Qn @ Tn.T):
            top = row.topk(3)
            print(f"  {k}: " + ", ".join(f"{names[i]}({v:.2f})" for v, i in zip(top.values, top.indices)), flush=True)

print(f"\nDONE in {time.time()-t0:.1f}s", flush=True)
