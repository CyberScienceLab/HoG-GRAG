import os
import math
import tiktoken
from openai import OpenAI
import pandas as pd
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
enc = tiktoken.get_encoding("cl100k_base")

def perplexity(text: str, model: str) -> float:
    
    resp = client.completions.create(
        model=model,
        prompt=text,
        max_tokens=0,       
        temperature=0,
        logprobs=1,         
        echo=True           
    )

    token_logprobs = resp.choices[0].logprobs.token_logprobs
    token_logprobs = [lp for lp in token_logprobs if lp is not None]

    if not token_logprobs:
        raise RuntimeError("No logprobs returned.")

    avg_nll = -sum(token_logprobs) / len(token_logprobs)
    return math.exp(avg_nll)


def add_perplexity_scores(
    csv_in: str,
    csv_out: str,
    text_col: str = "text",
    label_col: str = "label",
    model: str = "gpt-3.5-turbo-instruct"
):
    df = pd.read_csv(csv_in)
    ppls = []
    for txt in tqdm(df[text_col].astype(str), desc="Computing perplexity"):
        ppls.append(perplexity(txt, model=model))

    df["perplexity"] = ppls
    df.to_csv(csv_out, index=False)
    return df


def plot_roc_from_perplexity(
    df: pd.DataFrame,
    label_col: str = "label",
    score_col: str = "perplexity",
    title: str = "HotpotQA"  
):
    y_true = df[label_col].values        
    scores = df[score_col].values       

    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 9

    plt.figure(figsize=(2.5, 2.5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return fpr, tpr, thresholds, roc_auc


def build_n_clean_poison_csv(
    clean_txt_path: str,
    poison_txt_path: str,
    output_csv_path: str,
    n: int = 100,
    seed: int = 42
):
    random.seed(seed)

    with open(clean_txt_path, "r", encoding="utf-8") as f:
        clean_lines = [line.strip() for line in f if line.strip()]
    
    with open(poison_txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    poison_blocks = []
    buffer = []

    for line in lines:
        stripped = line.strip()
        if stripped:
            buffer.append(stripped)
            if len(buffer) == 2:  
                poison_blocks.append(" ".join(buffer))
                buffer = []
        else:
            buffer = []  
   
    clean_sample = random.sample(clean_lines, n)
    poison_sample = random.sample(poison_blocks, n)

    rows = (
        [{"text": t, "label": 0} for t in clean_sample] +
        [{"text": t, "label": 1} for t in poison_sample]
    )
    random.shuffle(rows)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False)
    return df

