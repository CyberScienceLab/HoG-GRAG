import pandas as pd
from openai import OpenAI
import os
import csv

my_api_key = os.getenv("OPENAI_KEY")
client = OpenAI(api_key=my_api_key)

def _extract_text_from_response(response) -> str:
    for out in response.output:
        if out.content:
            part = out.content[0]
            text = getattr(part, "text", None)
            if text is not None:
                return text.strip()
    return ""


def paraphrase_questions_from_csv(
    qid_csv: str,
    questions_csv: str,
    prompt_md: str,
    output_csv: str,
    qid_col: str = "qid",
    question_col: str = "question",
    model: str = "gpt-5-mini"
):
    qid_df = pd.read_csv(qid_csv)
    questions_df = pd.read_csv(questions_csv)

    merged = qid_df[[qid_col]].merge(
        questions_df[[qid_col, question_col]],
        on=qid_col,
        how="left"
    )

    with open(prompt_md, "r", encoding="utf-8") as f:
        base_prompt = f.read()

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([qid_col, question_col, "paraphrased"])

    for i, (_, row) in enumerate(merged.iterrows()):
        print(i)
        # if i==0:
        #     continue
        qid = row[qid_col]
        q = row[question_col]

        if pd.isna(q):
            paraphrased = ""
        else:
            prompt = f"{base_prompt}\n\nQuestion:\n{q}"
            response = client.responses.create(model=model, input=prompt)
            text = _extract_text_from_response(response)

            # Clean prefix
            lower = text.lower()
            if lower.startswith("paraphrased:"):
                text = text[len("paraphrased:"):].strip()

            paraphrased = text

        # Append immediately
        with open(output_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([qid, q, paraphrased])
