import code
import os
from pathlib import Path
import pandas as pd
from openai import OpenAI
import matplotlib.pyplot as plt
import json
import numpy as np
import ast
import re


def poison_reduction_gain(original_csv, repaired_csv):
    o = pd.read_csv(original_csv)
    r = pd.read_csv(repaired_csv)
    df = o.merge(r, on="qid", suffixes=("_orig", "_rep"))

    df["E_orig"] = df["num_poison_entities_orig"]
    df["E_rep"]  = df["num_poison_entities_rep"]
    df["R_orig"] = df["num_poison_relations_orig"]
    df["R_rep"]  = df["num_poison_relations_rep"]

    df_E = df[df["E_orig"] > 0].copy()
    df_R = df[df["R_orig"] > 0].copy()

    df_E["ratio_E"] = df_E["E_rep"] / df_E["E_orig"]
    df_R["ratio_R"] = df_R["R_rep"] / df_R["R_orig"]

    PRG_E = 1 - df_E["ratio_E"].mean() if not df_E.empty else float("nan")
    PRG_R = 1 - df_R["ratio_R"].mean() if not df_R.empty else float("nan")

    df["P_orig"] = df["E_orig"] + df["R_orig"]
    df["P_rep"]  = df["E_rep"]  + df["R_rep"]

    df_P = df[df["P_orig"] > 0].copy()  
    df_P["ratio_P"] = df_P["P_rep"] / df_P["P_orig"]
    PRG = 1 - df_P["ratio_P"].mean() if not df_P.empty else float("nan")

    return {
        "PRG": PRG,                 
        "PRG_E": PRG_E,
        "PRG_R": PRG_R,
        "per_query_combined": df_P[["qid", "P_orig", "P_rep", "ratio_P"]],
        "per_query_entities": df_E[["qid","E_orig","E_rep","ratio_E"]],
        "per_query_relations": df_R[["qid","R_orig","R_rep","ratio_R"]],
    }


def summarize_subgraph_before_repair(
    input_csv: str,
    output_csv: str,
    entity_poison_threshold: int ,
    relation_poison_threshold: int,
    entities_col: str = "entities",
    relations_col: str = "relationships",
):
    df = pd.read_csv(input_csv)
    rows = []
    for _, r in df.iterrows():
        qid = r.get("qid", "")
        ent_ids = _parse_id_list(r.get(entities_col))
        rel_ids = _parse_id_list(r.get(relations_col))
        n_entities = len(ent_ids)
        n_relations = len(rel_ids)
        n_poison_entities = sum(1 for eid in ent_ids if eid > entity_poison_threshold)
        n_poison_relations = sum(1 for rid in rel_ids if rid > relation_poison_threshold)
        n_benign_entities = n_entities - n_poison_entities
        n_benign_relations = n_relations - n_poison_relations
        rows.append(
            {
                "qid": qid,
                "num_entities": n_entities,
                "num_relations": n_relations,
                "num_benign_entities": n_benign_entities,
                "num_benign_relations": n_benign_relations,
                "num_poison_entities": n_poison_entities,
                "num_poison_relations": n_poison_relations,
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False)
    return out_df


def summarize_graph_after_repair(
    input_csv: str,
    repaired_folder: str,
    output_csv: str,
    entity_poison_threshold: int,
    relation_poison_threshold: int ,
    ent_file_suffix: str = "_entities_repaired.parquet",
    rel_file_suffix: str = "_relations_repaired.parquet",
    entity_id_col: str = "human_readable_id",
    relation_id_col: str = "human_readable_id",
):
    df = pd.read_csv(input_csv).query("llm_flag == 1")
    rows = []
    for _, r in df.iterrows():
        qid = r.get("qid", "")
        if not qid:
            continue
        ent_path = os.path.join(repaired_folder, f"q{qid}{ent_file_suffix}")
        rel_path = os.path.join(repaired_folder, f"q{qid}{rel_file_suffix}")
        if not os.path.exists(ent_path) or not os.path.exists(rel_path):
            continue

        ent_df = pd.read_parquet(ent_path)
        rel_df = pd.read_parquet(rel_path)

        if entity_id_col in ent_df.columns:
            ent_ids = ent_df[entity_id_col]
        else:
            int_cols = ent_df.select_dtypes(include="int").columns
            if len(int_cols) == 0:
                continue
            ent_ids = ent_df[int_cols[0]]

        if relation_id_col in rel_df.columns:
            rel_ids = rel_df[relation_id_col]
        else:
            int_cols = rel_df.select_dtypes(include="int").columns
            if len(int_cols) == 0:
                continue
            rel_ids = rel_df[int_cols[0]]

        ent_ids = ent_ids.dropna().astype(int).tolist()
        rel_ids = rel_ids.dropna().astype(int).tolist()

        n_entities = len(ent_ids)
        n_relations = len(rel_ids)

        n_poison_entities = sum(1 for eid in ent_ids if eid > entity_poison_threshold)
        n_poison_relations = sum(1 for rid in rel_ids if rid > relation_poison_threshold)

        n_benign_entities = n_entities - n_poison_entities
        n_benign_relations = n_relations - n_poison_relations

        rows.append(
            {
                "qid": qid,
                "num_entities": n_entities,
                "num_relations": n_relations,
                "num_benign_entities": n_benign_entities,
                "num_benign_relations": n_benign_relations,
                "num_poison_entities": n_poison_entities,
                "num_poison_relations": n_poison_relations,
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False)
    return out_df


def judge_QA_with_llm(input_csv, output_csv, prompt_md_path, model_name="gpt-5-mini"):
    with open(prompt_md_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()
    df = pd.read_csv(input_csv)

    def judge_row(row):
        q = str(row.get("question", ""))
        gold = str(row.get("final_answer", ""))  
        model_ans = str(row.get("model_answer", "")) 
        if model_ans.strip() == "":
            return 0
        user_content = (
            "Decide whether the model_answer correctly answers the question, "
            "according to the gold_answer.\n\n"
            f"Question: {q}\n"
            f"Gold answer: {gold}\n"
            f"Model answer: {model_ans}\n\n"
            'Respond only with JSON: {"flag": 1} or {"flag": 0}.'
        )
        r = client.responses.create(
            model=model_name,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        text = ""
        if getattr(r, "output", None):
            for item in r.output:
                parts = getattr(item, "content", None) or []
                for p in parts:
                    t = getattr(p, "text", None)
                    if t:
                        text += t
        text = text.strip()
        try:
            flag = int(json.loads(text).get("flag", 0))
        except:
            flag = 0
        print(f"gold: {gold} model: {model_ans} -> flag: {flag}")
        return flag

    df["llm_flag"] = df.apply(judge_row, axis=1)
    df.to_csv(output_csv, index=False)


def subgraph_compactness(
    original_csv: str,
    repaired_csv: str,
    qid_col: str = "qid",
    ent_col: str = "num_entities",
    rel_col: str = "num_relations",
):
    df_orig = pd.read_csv(original_csv)
    df_rep = pd.read_csv(repaired_csv)

    df = df_orig[[qid_col, ent_col, rel_col]].merge(
        df_rep[[qid_col, ent_col, rel_col]],
        on=qid_col,
        suffixes=("_orig", "_rep"),
        how="inner",
    )
    df["size_orig"] = df[f"{ent_col}_orig"] + df[f"{rel_col}_orig"]
    df["size_rep"] = df[f"{ent_col}_rep"] + df[f"{rel_col}_rep"]
    df = df[df["size_orig"] > 0].copy()
    df["sc_ratio"] = df["size_rep"] / df["size_orig"]
    sc_value = df["sc_ratio"].mean()
    return sc_value, df[[qid_col, "size_orig", "size_rep", "sc_ratio"]]
