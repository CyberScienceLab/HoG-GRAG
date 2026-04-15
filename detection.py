import ast
from importlib_metadata import metadata
import pandas as pd
from pathlib import Path
import os
from openai import OpenAI
import re
import tempfile
import pandas as pd
import json
import ast

my_api_key = os.getenv("OPENAI_KEY")
client = OpenAI(api_key=my_api_key)


def get_subqueries_by_qid(csv_path: str, qid: str):
    df = pd.read_csv(csv_path)
    row = df.loc[df["qid"] == qid]
    if row.empty:
        raise ValueError(f"qid not found: {qid}")
    raw = row.iloc[0]["decomposed_questions"]
    dq = None
    if isinstance(raw, str):
        s = raw.strip()
        try:
            dq = json.loads(s)
        except Exception:
            try:
                dq = ast.literal_eval(s)
            except Exception:
                raise ValueError("Could not parse decomposed_questions as JSON or Python literal.")
    else:
        dq = raw

    if not isinstance(dq, list) or len(dq) < 2:
        raise ValueError("decomposed_questions must be a list with at least 2 items.")

    q1 = dq[0].get("subquestion")
    q2 = dq[1].get("subquestion")

    if not q1 or not q2:
        raise ValueError("Missing subquestion fields in decomposed_questions.")

    return q1, q2


def build_context_text(context):
    entity_frames = []
    relation_frames = []
    for item in context:
        if isinstance(item, (list, tuple)):
            entities_file = item[0] if len(item) > 1 else None
            relations_file = item[-1]
        else:
            entities_file = None
            relations_file = item

        if entities_file and isinstance(entities_file, str) and entities_file.endswith(".parquet") and os.path.exists(entities_file):
            entity_frames.append(pd.read_parquet(entities_file))

        if relations_file and isinstance(relations_file, str) and relations_file.endswith(".parquet") and os.path.exists(relations_file):
            relation_frames.append(pd.read_parquet(relations_file))

    entity_texts = []
    if entity_frames:
        df = pd.concat(entity_frames, ignore_index=True).drop_duplicates("title")
        entity_texts = [f"{r.title}: {r.get('description','')}" for _, r in df.iterrows()]

    relation_texts = []
    if relation_frames:
        df = pd.concat(relation_frames, ignore_index=True).drop_duplicates(["source","target","description"])
        relation_texts = [f"{r.source} —({r.description})→ {r.target}" for _, r in df.iterrows()]

    return "\n".join(entity_texts + relation_texts)


def get_answer_subquestion(context, subquestion):
    context = build_context_text(context)  
    with open("prompts/response_subquestion-detection.md", "r", encoding="utf-8") as f:
        template = f.read()
    prompt = template.format(
        context=context,
        question=subquestion
    )
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.choices[0].message.content.strip()
    return answer


def detection(QA_file, decomposed_questions_file, output_file):
    df = pd.read_csv(QA_file, encoding="utf-8")
    for idx, row in df.iterrows():
        print(idx)
        qid_raw = row.get("qid")        
        qid = str(qid_raw).strip()
        print(f"Processing QID: {qid}")
        q1, q2 = get_subqueries_by_qid_hotpot(decomposed_questions_file, qid)
        context = [(f"q{qid}_entities.parquet",f"q{qid}_relations.parquet")]
        ans1 = (get_answer_subquestion(context, q1).strip().replace('"', '').replace("'", ''))
        detect_flag = 0
        ans2 = "-"
        if ans1.upper() == "NA":
            ans1 = ans1.upper()
            detect_flag = 11
        elif ";" in ans1:
            detect_flag = 12
        else:
            q2_filled = re.sub(r'\[.*?\]', f'[{ans1}]', q2)
            ans2 = (get_answer_subquestion(context, q2_filled).strip().replace('"', '').replace("'", ''))
            # print(ans2)
            if ans2.upper() == "NA":
                ans2 = ans2.upper()
                detect_flag = 21
            elif ";" in ans2:
                detect_flag = 22
            else:
                detect_flag = 23            
        print(f"Detect flag: {detect_flag}")
        row_out = pd.DataFrame([[
            qid,
            q1,
            q2,
            ans1,
            ans2,            
            detect_flag
        ]], columns=[
            "qid",
            "q1",
            "q2",
            "ans1",
            "ans2",            
            "detect_flag"
        ])
        row_out.to_csv(
            output_file,
            mode='a',
            header=not os.path.exists(output_file),
            index=False
        )
     
