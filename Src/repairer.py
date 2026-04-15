import ast
from importlib_metadata import metadata
import pandas as pd
from pathlib import Path
import os
from openai import OpenAI
import tempfile
import re
import json
from typing import List, Optional, Dict, Any, Tuple
from typing import Set, Tuple, Dict, Any, List
import unicodedata
import numpy as np
import torch
from typing import Any, Dict, List, Set, Tuple, Optional
RelKey = Tuple[str, str, str]
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import torch
my_api_key = os.getenv("OPENAI_KEY")
client = OpenAI(api_key=my_api_key)
_SLOT_PAT = re.compile(r"#1")


def Fill(subquery: str, prev_answer: Optional[str]) -> str:
    if prev_answer is None:
        return subquery
    prev_answer = str(prev_answer).strip()
    if not prev_answer or prev_answer == "⊥":
        return subquery
    return _SLOT_PAT.sub(prev_answer, subquery, count=1)


def parse_llm_answer_per_answer_evidence(output_text: str) -> Tuple[int, List[str], List[List[str]]]:
    s = output_text.strip()
    if s.strip().strip('"').strip("'").upper() == "NA":
        return 0, [], []
    if "|" not in s:
        raise ValueError(f"Invalid format (missing '|'): {s}")
    answer_part, evidence_part = s.split("|", 1)
    answers_raw = [a.strip() for a in answer_part.split(";") if a.strip()]
    groups = re.findall(r"\[(.*?)\]", evidence_part)
    if not groups:
        raise ValueError(f"Invalid evidence format (no [..] groups): {s}")
    evidence_lists = []
    for grp in groups:
        ids = [x.strip() for x in grp.split(",") if x.strip()]
        evidence_lists.append(ids)
    if len(evidence_lists) == 1 and len(answers_raw) > 1:
        evidence_lists = evidence_lists * len(answers_raw)
    if len(evidence_lists) != len(answers_raw):
        raise ValueError(f"Mismatch: {len(answers_raw)} answers but {len(evidence_lists)} evidence groups: {s}")
    answers = []
    filtered_evidence = []
    for a, ev in zip(answers_raw, evidence_lists):
        a_norm = a.strip().strip('"').strip("'").upper()
        if a_norm == "NA":
            continue
        answers.append(a)
        filtered_evidence.append(ev)
    if not answers:
        return 0, [], []
    return len(answers), answers, filtered_evidence


def build_context_text_from_dfs(ent_df: pd.DataFrame, rel_df: pd.DataFrame) -> Tuple[str, List[Dict[str, Any]]]:
    lines = []
    evidence_index = []
    if ent_df is not None and len(ent_df) > 0:
        if "title" not in ent_df.columns:
            raise ValueError("Entity dataframe must contain 'title'.")
        if "description" not in ent_df.columns:
            ent_df = ent_df.copy()
            ent_df["description"] = ""

        ent_df = ent_df.drop_duplicates("title")
        eid = 1
        for _, row in ent_df.iterrows():
            ev_id = f"E{eid}"
            title = str(row["title"])
            desc = str(row.get("description", ""))
            lines.append(f"[{ev_id}] {title}: {desc}".strip())
            evidence_index.append({"id": ev_id, "type": "entity", "title": title, "description": desc})
            eid += 1

    if rel_df is not None and len(rel_df) > 0:
        need = {"source", "target", "description"}
        if not need.issubset(rel_df.columns):
            raise ValueError("Relation dataframe must contain: source, target, description")

        rel_df = rel_df.drop_duplicates(["source", "target", "description"])
        rid = 1
        for _, row in rel_df.iterrows():
            ev_id = f"R{rid}"
            s = str(row["source"])
            t = str(row["target"])
            d = str(row["description"])
            lines.append(f"[{ev_id}] {s} —({d})→ {t}".strip())
            evidence_index.append({"id": ev_id, "type": "relation", "source": s, "target": t, "relation": d})
            rid += 1
    return "\n".join(lines), evidence_index


def id_index_map(evidence_index: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {rec["id"]: rec for rec in evidence_index if "id" in rec}

def materialize_evidence(evidence_ids: List[str], idxmap: Dict[str, Dict[str, Any]]) -> Tuple[Set[str], Set[Tuple[str,str,str]]]:
    ent_titles = set()
    rel_keys = set()
    for evid in evidence_ids:
        rec = idxmap.get(evid)
        if not rec:
            continue
        if rec.get("type") == "entity":
            ent_titles.add(rec.get("title", ""))
        elif rec.get("type") == "relation":
            rel_keys.add((rec.get("source", ""), rec.get("target", ""), rec.get("relation", "")))
    ent_titles.discard("")
    rel_keys.discard(("", "", ""))
    return ent_titles, rel_keys


def inject_evidence(ent_work: pd.DataFrame, rel_work: pd.DataFrame,
                    ent_add: pd.DataFrame, rel_add: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if ent_add is not None and len(ent_add) > 0:
        ent_work = pd.concat([ent_work, ent_add], ignore_index=True).drop_duplicates("title")
    if rel_add is not None and len(rel_add) > 0:
        rel_work = pd.concat([rel_work, rel_add], ignore_index=True).drop_duplicates(["source","target","description"])
    return ent_work, rel_work


def remove_evidence(ent_work: pd.DataFrame, rel_work: pd.DataFrame, ent_titles: Set[str], rel_keys: Set[Tuple[str,str,str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if ent_titles:
        ent_work = ent_work[~ent_work["title"].isin(ent_titles)].copy()
    if rel_keys:
        keep_mask = []
        for _, row in rel_work.iterrows():
            k = (str(row["source"]), str(row["target"]), str(row["description"]))
            keep_mask.append(k not in rel_keys)
        rel_work = rel_work[pd.Series(keep_mask, index=rel_work.index)].copy()
    return ent_work, rel_work


def rollback_from(j: int, H: int, ent_work: pd.DataFrame, rel_work: pd.DataFrame,
                  r: list, E: list, E_ids: list) -> Tuple[pd.DataFrame, pd.DataFrame, list, list, list]:

    for t in range(H, j-1, -1):
        if E[t]:
            ent_titles_t, rel_keys_t = E[t]
            ent_work, rel_work = remove_evidence(ent_work, rel_work, ent_titles_t, rel_keys_t)
        r[t] = None
        E[t] = None
        E_ids[t] = None
    return ent_work, rel_work, r, E, E_ids


def get_answer_subquestion(context, subquestion):
    with open("prompts/response_subquestion.md", "r", encoding="utf-8") as f:
        template = f.read()
    prompt = template.format(
        context=context,
        subquestion=subquestion
    )
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.choices[0].message.content.strip()
    return answer


def answer_over_graph(subquery: str, ent_df: pd.DataFrame, rel_df: pd.DataFrame) -> Tuple[List[str], Dict[str, List[str]], Dict[str, Dict[str, Any]]]:
    context_text, evidence_index = build_context_text_from_dfs(ent_df, rel_df)

    raw = get_answer_subquestion(context_text, subquery)  
    raw = raw.strip().replace('"', '').replace("'", "")
    num, answers, evidence_lists = parse_llm_answer_per_answer_evidence(raw)
    idxmap = id_index_map(evidence_index)
    if num == 0:
        return [], {}, idxmap
    Phi = {a: ids for a, ids in zip(answers, evidence_lists)}
    return answers, Phi, idxmap


def _norm_text(x: str) -> str:
    s = "" if x is None else str(x)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = re.sub(r"[^0-9a-z]+", " ", s)     
    s = re.sub(r"\s+", " ", s).strip()     
    return s


def entities_from_evidence_ids(
    evidence_ids: List[str],
    idxmap: Dict[str, Dict[str, Any]],
) -> Set[str]:

    ent_set: Set[str] = set()
    for evid in evidence_ids:
        rec = idxmap.get(evid)
        if not rec:
            continue
        rtype = rec.get("type")
        if rtype == "entity":
            t = _norm_text(rec.get("title", ""))
            if t:
                ent_set.add(t)
        elif rtype == "relation":
            s = _norm_text(rec.get("source", ""))
            t = _norm_text(rec.get("target", ""))
            if s:
                ent_set.add(s)
            if t:
                ent_set.add(t)
    return ent_set


def prefix_entities_from_prefix_evidence(E_prefix: Tuple[Set[str], Set[RelKey]]) -> Set[str]:
    prefix_ents_raw, prefix_rels_raw = E_prefix
    prefix_ent_set: Set[str] = set()
    for e in prefix_ents_raw:
        ne = _norm_text(e)
        if ne:
            prefix_ent_set.add(ne)
    for (s, t, _r) in prefix_rels_raw:
        ns = _norm_text(s)
        nt = _norm_text(t)
        if ns:
            prefix_ent_set.add(ns)
        if nt:
            prefix_ent_set.add(nt)
    return prefix_ent_set


def choose_last_hop_by_entity_intersection(
    cands: List[str],
    Phi: Dict[str, List[str]],
    idxmap: Dict[str, Dict[str, Any]],
    E_prefix: Tuple[Set[str], Set[RelKey]],):

    prefix_ent_set = prefix_entities_from_prefix_evidence(E_prefix)
    best = None
    best_key = None
    best_ent_set = None
    details = []
    for a in cands:
        ev_ids = Phi.get(a, [])
        cand_ent_set = entities_from_evidence_ids(ev_ids, idxmap)
        ent_ov = len(cand_ent_set & prefix_ent_set)
        details.append({
            "candidate": a,
            "ent_overlap": ent_ov,
            "prefix_ent_size": len(prefix_ent_set),
            "cand_ent_size": len(cand_ent_set),
            "evidence_len": len(ev_ids),
        })
        key = (ent_ov, -len(ev_ids), a)
        if best is None or key > best_key:
            best = a
            best_key = key
            best_ent_set = cand_ent_set
    return best, best_ent_set, details


def repair_one_question(qid: str,
                        q_list_1indexed: List[Optional[str]],
                        Gq_entities_file: str,
                        Gq_relations_file: str,
                        all_entities_file: str,
                        all_relations_file: str,
                        H: int ):

    ent_work = pd.read_parquet(Gq_entities_file)
    rel_work = pd.read_parquet(Gq_relations_file)
    r = [None] * (H + 1)      
    E = [None] * (H + 1)     
    E_ids = [None] * (H + 1)  
    S: List[Dict[str, Any]] = []
    trace: List[Dict[str, Any]] = []
    tried: Dict[Tuple[int, tuple], set] = {}
    def log(ev: str, **kw):
        rec = {"event": ev, **kw}
        trace.append(rec)
    def _inject_from_kg(ent_kg: pd.DataFrame,
                        rel_kg: pd.DataFrame,
                        ev_ids: List[str],
                        idxmap: Dict[str, Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame, Tuple[set, set]]:
        ent_titles, rel_keys = materialize_evidence(ev_ids, idxmap)

        ent_add = ent_kg[ent_kg["title"].isin(ent_titles)].copy() if ent_titles else ent_kg.iloc[0:0].copy()
        if rel_keys:
            rel_add = rel_kg[rel_kg.apply(
                lambda x: (str(x["source"]), str(x["target"]), str(x["description"])) in rel_keys, axis=1
            )].copy()
        else:
            rel_add = rel_kg.iloc[0:0].copy()
        ew, rw = inject_evidence(ent_work, rel_work, ent_add, rel_add)
        return ew, rw, (ent_titles, rel_keys)

                            def _prune_stack_from(hop_idx: int):
        nonlocal S
        S = [cp for cp in S if cp["hop"] < hop_idx]

    i = 1
    while i <= H:
        subq = q_list_1indexed[i]
        q_star = Fill(subq, r[i-1])
        log("hop_start", hop=i, subquery=subq, filled=q_star, prev_answer=r[i-1], stack_size=len(S))
        C, Phi, idxmap = answer_over_graph(q_star, ent_work, rel_work)
        log("work_answer", hop=i, num_candidates=len(C), candidates=C)
        if C:
            if i == H and len(C) > 1:
                E_pref = union_prefix_evidence(E, H - 1)
                log("last_hop_overlap_start",
                    hop=i, source="WORK",
                    num_candidates=len(C),
                    prefix_ent_size=len(E_pref[0]),
                    prefix_rel_size=len(E_pref[1]))
                chosen, cand_ent_set, cand_details = choose_last_hop_by_entity_intersection(
                    C, Phi, idxmap, E_pref
                )
                log("last_hop_overlap_candidates",
                    hop=i, source="WORK",
                    candidates=cand_details)
                r[i] = chosen
                E_ids[i] = Phi[chosen]
                E[i] = materialize_evidence(E_ids[i], idxmap)
                log("last_hop_overlap_commit",
                    hop=i, source="WORK",
                    chosen=chosen,
                    overlap_len = next(d["ent_overlap"] for d in cand_details if d["candidate"] == chosen),
                    evidence_ids=E_ids[i])
                i += 1
                continue
            else:
                chosen = C[0]
                remaining = C[1:]
                if remaining:
                    S.append({"hop": i, "remaining": remaining, "Phi": Phi, "idxmap": idxmap, "source": "WORK"})
                    log("push_choicepoint", hop=i, source="WORK", remaining=remaining, stack_size=len(S))
                r[i] = chosen
                E_ids[i] = Phi[chosen]
                E[i] = materialize_evidence(E_ids[i], idxmap)
                log("commit", hop=i, source="WORK", chosen=chosen, evidence_ids=E_ids[i])
                i += 1
                continue

        ent_kg, rel_kg =  get_kg_context_subset_semantic(kg_idx, q_star)
        log("kg_context", hop=i,  kg_entities=len(ent_kg), kg_relations=len(rel_kg))
        Ck, Phik, idxmap_k = answer_over_graph(q_star, ent_kg, rel_kg)
        log("kg_answer", hop=i, num_candidates=len(Ck), candidates=Ck)
        if Ck:
            if i == H and len(Ck) > 1:
                E_pref = union_prefix_evidence(E, H - 1)
                log("last_hop_overlap_start",
                    hop=i, source="KG",
                    num_candidates=len(Ck),
                    prefix_ent_size=len(E_pref[0]),
                    prefix_rel_size=len(E_pref[1]))
                chosen, cand_ent_set, cand_details = choose_last_hop_by_entity_intersection(
                    Ck, Phik, idxmap_k, E_pref)

                log("last_hop_overlap_candidates", hop=i, source="KG", candidates=cand_details)
                r[i] = chosen
                E_ids[i] = Phik[chosen]
                E[i] = materialize_evidence(E_ids[i], idxmap_k)
                ent_work, rel_work, ev_mat = _inject_from_kg(ent_kg, rel_kg, E_ids[i], idxmap_k)
                
                log("last_hop_overlap_commit",
                    hop=i, source="WORK",
                    chosen=chosen,
                    overlap_len = next(d["ent_overlap"] for d in cand_details if d["candidate"] == chosen),
                    evidence_ids=E_ids[i])
                i += 1
                continue
            else:
                chosen = Ck[0]
                remaining = Ck[1:]
                if remaining:
                    S.append({
                        "hop": i, "remaining": remaining, "Phi": Phik, "idxmap": idxmap_k, "source": "KG",
                        "kg_entities": ent_kg, "kg_relations": rel_kg
                    })
                    log("push_choicepoint", hop=i, source="KG", remaining=remaining, stack_size=len(S))
                r[i] = chosen
                E_ids[i] = Phik[chosen]
                ent_work, rel_work, ev_mat = _inject_from_kg(ent_kg, rel_kg, E_ids[i], idxmap_k)
                E[i] = ev_mat
                log("commit", hop=i, source="KG", chosen=chosen, evidence_ids=E_ids[i], injected=True)
                i += 1
                continue

        log("hop_infeasible", hop=i)
        if S:
            cp = S.pop()
            j = cp["hop"]
            log("backtrack_choicepoint", from_hop=i, to_hop=j, popped_source=cp["source"], stack_size=len(S))
            ent_work, rel_work, r, E, E_ids = rollback_from(j, H, ent_work, rel_work, r, E, E_ids)
            _prune_stack_from(j)
            log("rollback", rollback_from_hop=j)
            remaining = cp["remaining"]
            if not remaining:
                log("choicepoint_empty", hop=j)
                i = j
                continue
            nxt = remaining[0]
            rest = remaining[1:]
            if rest:
                cp["remaining"] = rest
                S.append(cp)
                log("choicepoint_update", hop=j, remaining=rest, stack_size=len(S))
            r[j] = nxt
            Phi_j = cp["Phi"]
            idxmap_j = cp["idxmap"]
            E_ids[j] = Phi_j[nxt]
            E[j] = materialize_evidence(E_ids[j], idxmap_j)
            log("commit_after_backtrack", hop=j, chosen=nxt, source=cp["source"], evidence_ids=E_ids[j])
            if cp["source"] == "KG":
                ent_kg_j = cp["kg_entities"]
                rel_kg_j = cp["kg_relations"]
                ent_work, rel_work, ev_mat = _inject_from_kg(ent_kg_j, rel_kg_j, E_ids[j], idxmap_j)
                E[j] = ev_mat
                log("inject_after_backtrack", hop=j, injected=True)
            i = j + 1
            log("resume", next_hop=i)
            continue
        if i == 1:
            log("fail", reason="infeasible_at_hop1_no_stack")
            return False, r, ent_work, rel_work, trace
        j = i - 1
        log("revise_prev_start", current_hop=i, revise_hop=j)
        ent_work, rel_work, r, E, E_ids = rollback_from(j, H, ent_work, rel_work, r, E, E_ids)
        _prune_stack_from(j)
        log("rollback", rollback_from_hop=j)

        subq_prev = q_list_1indexed[j]
        q_prev_star = Fill(subq_prev, r[j-1])
        ent_kg_prev, rel_kg_prev =  get_kg_context_subset_semantic(kg_idx, q_prev_star)
        log("kg_context", hop=j, kg_entities=len(ent_kg_prev), kg_relations=len(rel_kg_prev))
        Cprev, Phiprev, idxmap_prev = answer_over_graph(q_prev_star, ent_kg_prev, rel_kg_prev)
        log("kg_answer", hop=j, num_candidates=len(Cprev), candidates=Cprev)
        if not Cprev:
            log("fail", reason="revise_prev_no_candidates", revise_hop=j)
            return False, r, ent_work, rel_work, trace

        pk_prev = tuple(r[1:j])  # prefix up to hop j-1 (since r[j] is None after rollback)
        key_prev = (j, pk_prev)
        tried_set = tried.setdefault(key_prev, set())
        chosen_prev = None
        for c in Cprev:
            if c not in tried_set:
                chosen_prev = c
                break
        if chosen_prev is None:
            log("fail", reason="revise_prev_exhausted_candidates", revise_hop=j)
            return False, r, ent_work, rel_work, trace
        tried_set.add(chosen_prev)
        remaining_prev = [c for c in Cprev if c != chosen_prev]
        if remaining_prev:
            S.append({
                "hop": j, "remaining": remaining_prev, "Phi": Phiprev, "idxmap": idxmap_prev, "source": "KG",
                "kg_entities": ent_kg_prev, "kg_relations": rel_kg_prev
            })
            log("push_choicepoint", hop=j, source="KG", remaining=remaining_prev, stack_size=len(S))

        r[j] = chosen_prev
        E_ids[j] = Phiprev[chosen_prev]
        ent_work, rel_work, ev_mat = _inject_from_kg(ent_kg_prev, rel_kg_prev, E_ids[j], idxmap_prev)
        E[j] = ev_mat
        log("revise_prev_commit", hop=j, chosen=chosen_prev, evidence_ids=E_ids[j], injected=True)
        i = j + 1
        log("resume", next_hop=i)
        continue

    ent_hat, rel_hat = induce_graph(E, ent_work, rel_work)
    log("induce_graph", induced_entities=len(ent_hat), induced_relations=len(rel_hat))
    log("success", hop_answers=r[1:H+1])
    return True, r, ent_hat, rel_hat, trace


def induce_graph(E: List[Tuple[Set[str], Set[Tuple[str,str,str]]]],
                 ent_work: pd.DataFrame,
                 rel_work: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    keep_titles = set()
    keep_relkeys = set()
    for ev in E:
        if not ev:
            continue
        titles, relkeys = ev
        keep_titles.update(titles)
        keep_relkeys.update(relkeys)
    ent_small = ent_work[ent_work["title"].isin(keep_titles)].drop_duplicates("title").copy() if keep_titles else ent_work.iloc[0:0].copy()
    if keep_relkeys:
        rel_small = rel_work[
            rel_work.apply(lambda x: (str(x["source"]), str(x["target"]), str(x["description"])) in keep_relkeys, axis=1)
        ].drop_duplicates(["source","target","description"]).copy()
    else:
        rel_small = rel_work.iloc[0:0].copy()
    return ent_small, rel_small


def union_prefix_evidence(E: List[Any], upto_hop: int) -> Tuple[Set[str], Set[RelKey]]:
    ent_u: Set[str] = set()
    rel_u: Set[RelKey] = set()
    for t in range(1, upto_hop + 1):
        if not E[t]:
            continue
        ent_t, rel_t = E[t]
        ent_u |= set(ent_t)
        rel_u |= set(rel_t)
    return ent_u, rel_u


def repair(input_file: str,
           all_entities_file: str,
           all_relations_file: str,
           decomposed_questions_file: str,
           subgraph_dir: str,
           out_dir: str,
           results_csv: str,
           H: int = 2):

    df = pd.read_csv(input_file, encoding="utf-8")
    results = []
    i = 0
    for _, row in df.iterrows():
        i = i + 1
        print(i)
        qid = row.get("qid")
        q1, q2 = get_subquestions_hotpot(decomposed_questions_file, qid)
        q_list = [None, q1, q2]  
        ent_file = os.path.join(subgraph_dir, f"q{qid}_entities.parquet")
        rel_file = os.path.join(subgraph_dir, f"q{qid}_relations.parquet")
        if not (os.path.exists(ent_file) and os.path.exists(rel_file)):
            results.append({
                "qid": qid, "success": False, "reason": "missing_subgraph",
                "trace_json": json.dumps([{"event": "fail", "reason": "missing_subgraph"}], ensure_ascii=False)
            })
            continue

        ok, r, ent_work, rel_work, trace = repair_one_question(
            qid=qid,
            q_list_1indexed=q_list,
            Gq_entities_file=ent_file,
            Gq_relations_file=rel_file,
            all_entities_file=all_entities_file,
            all_relations_file=all_relations_file,
            H=H
        )
        if ok:
            out_ent = os.path.join(out_dir, f"q{qid}_entities_repaired.parquet")
            out_rel = os.path.join(out_dir, f"q{qid}_relations_repaired.parquet")
            ent_work.to_parquet(out_ent, index=False)
            rel_work.to_parquet(out_rel, index=False)
        row_result = {
            "qid": qid,
            "success": bool(ok),
            "hop_answers": ";".join([x for x in (r[1:H+1]) if x]) if ok else "",
            "trace_json": json.dumps(trace, ensure_ascii=False)
            }
        pd.DataFrame([row_result]).to_csv(
            results_csv,
            mode="a",           # append
            header=False,       # header already written
            index=False,
            encoding="utf-8"
            )


def get_subquestions(csv_path: str, qid: str):
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


@dataclass
class KGSemanticIndex:
    ent_all: pd.DataFrame
    rel_all: pd.DataFrame
    ent_text: list[str]
    rel_text: list[str]
    ent_emb: np.ndarray
    rel_emb: np.ndarray
    model_name: str
    device: str

_KG_INDEX_CACHE: Dict[Tuple[str, str, str], KGSemanticIndex] = {}
_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}


def _get_model(model_name: str, device: str):
    key = (model_name, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)
    _MODEL_CACHE[key] = model
    return model


def _embed_openai(
    texts,
    model: str,
    batch_size: int = 256,
):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(
            model=model,
            input=batch
        )
        all_embs.extend([d.embedding for d in resp.data])
    emb = np.asarray(all_embs, dtype=np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return emb


def _embed_query_openai(q: str, model: str) -> np.ndarray:
    resp = client.embeddings.create(model=model, input=[q])
    v = np.asarray(resp.data[0].embedding, dtype=np.float32)
    v = v / np.linalg.norm(v)
    return v

def get_kg_context_subset_semantic(
    kg_index: KGSemanticIndex,
    question: str,
    max_total: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    q = (question or "").strip()
    if not q:
        ne = min(max_total, len(kg_index.ent_all))
        nr = min(max_total - ne, len(kg_index.rel_all))
        return (
            kg_index.ent_all.head(ne).copy(),
            kg_index.rel_all.head(nr).copy(),
        )

    if str(kg_index.model_name).startswith("text-embedding-"):
        qv = _embed_query_openai(q, model=kg_index.model_name)
    else:
        model = _get_model(kg_index.model_name, device=kg_index.device)
        q_emb = model.encode([q], normalize_embeddings=True, batch_size=1, show_progress_bar=False)
        qv = np.asarray(q_emb[0], dtype=np.float32)

    ent_scores = (kg_index.ent_emb @ qv).astype(np.float32)
    rel_scores = (kg_index.rel_emb @ qv).astype(np.float32)

    ent_idx = np.arange(len(ent_scores), dtype=int)
    rel_idx = np.arange(len(rel_scores), dtype=int)

    all_types = np.concatenate([np.array(["E"] * len(ent_idx)), np.array(["R"] * len(rel_idx))])
    all_idx = np.concatenate([ent_idx, rel_idx])
    all_scores = np.concatenate([ent_scores, rel_scores])

    type_key = (all_types == "R").astype(int)  # E=0, R=1
    order = np.lexsort((all_idx, type_key, -all_scores))

    top = order[: min(max_total, len(order))]

    picked_ent = [int(all_idx[i]) for i in top if all_types[i] == "E"]
    picked_rel = [int(all_idx[i]) for i in top if all_types[i] == "R"]

    ent_sub = kg_index.ent_all.iloc[picked_ent].copy() if picked_ent else kg_index.ent_all.head(0).copy()
    rel_sub = kg_index.rel_all.iloc[picked_rel].copy() if picked_rel else kg_index.rel_all.head(0).copy()

    if ent_sub.empty and rel_sub.empty:
        ne = min(max_total, len(kg_index.ent_all))
        nr = min(max_total - ne, len(kg_index.rel_all))
        ent_sub = kg_index.ent_all.head(ne).copy()
        rel_sub = kg_index.rel_all.head(nr).copy()

    return ent_sub, rel_sub


def build_kg_semantic_index(
    all_entities_file: str,
    all_relations_file: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None,
    batch_size: int = 256,
    cache: bool = True,
) -> KGSemanticIndex:
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_key = (all_entities_file, all_relations_file, model_name)
    if cache and cache_key in _KG_INDEX_CACHE:
        idx = _KG_INDEX_CACHE[cache_key]
        if idx.device == device:
            return idx

    ent_all = pd.read_parquet(all_entities_file).copy()
    rel_all = pd.read_parquet(all_relations_file).copy()

    if "title" not in ent_all.columns:
        raise ValueError("Entities parquet must contain a 'title' column.")
    if "source" not in rel_all.columns or "target" not in rel_all.columns:
        raise ValueError("Relations parquet must contain 'source' and 'target' columns.")

    ent_text_s = ent_all["title"].fillna("").astype(str)
    if "description" in ent_all.columns:
        ent_text_s = ent_text_s + " " + ent_all["description"].fillna("").astype(str)
    ent_text = ent_text_s.tolist()

    rel_text_s = (
        rel_all["source"].fillna("").astype(str)
        + " "
        + rel_all["target"].fillna("").astype(str)
    )
    if "description" in rel_all.columns:
        rel_text_s = rel_text_s + " " + rel_all["description"].fillna("").astype(str)
    rel_text = rel_text_s.tolist()

    if model_name.startswith("text-embedding-"):
        ent_emb = _embed_openai(ent_text, model=model_name, batch_size=batch_size)
        rel_emb = _embed_openai(rel_text, model=model_name, batch_size=batch_size)
    else:
        model = _get_model(model_name, device=device)
        ent_emb = model.encode(
            ent_text,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        rel_emb = model.encode(
            rel_text,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )

    idx = KGSemanticIndex(
        ent_all=ent_all,
        rel_all=rel_all,
        ent_text=ent_text,
        rel_text=rel_text,
        ent_emb=np.asarray(ent_emb),
        rel_emb=np.asarray(rel_emb),
        model_name=model_name,
        device=device,
    )

    if cache:
        _KG_INDEX_CACHE[cache_key] = idx
    return idx


def save_kg_semantic_index(
    kg_index,
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)
    kg_index.ent_all.to_parquet(os.path.join(out_dir, "entities.parquet"))
    kg_index.rel_all.to_parquet(os.path.join(out_dir, "relations.parquet"))
    np.save(os.path.join(out_dir, "ent_emb.npy"), kg_index.ent_emb)
    np.save(os.path.join(out_dir, "rel_emb.npy"), kg_index.rel_emb)
    meta = {
        "model_name": kg_index.model_name,
        "device": kg_index.device,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_kg_semantic_index(
    cache_dir: str,
) -> KGSemanticIndex:
    cache_dir = Path(cache_dir).resolve()

    ent_all = pd.read_parquet(cache_dir / "entities.parquet")
    rel_all = pd.read_parquet(cache_dir / "relations.parquet")

    ent_emb = np.load(cache_dir / "ent_emb.npy")
    rel_emb = np.load(cache_dir / "rel_emb.npy")

    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json in {cache_dir}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    model_name = meta.get("model_name", "unknown")
    device = meta.get("device", "cpu")

    if ent_emb.ndim != 2 or rel_emb.ndim != 2:
        raise ValueError("Embedding arrays must be 2D")

    if len(ent_all) != ent_emb.shape[0]:
        raise ValueError(
            f"Entity count mismatch: {len(ent_all)} rows vs {ent_emb.shape[0]} embeddings"
        )

    if len(rel_all) != rel_emb.shape[0]:
        raise ValueError(
            f"Relation count mismatch: {len(rel_all)} rows vs {rel_emb.shape[0]} embeddings"
        )

    return KGSemanticIndex(
        ent_all=ent_all,
        rel_all=rel_all,
        ent_text=[],   # not needed after embedding
        rel_text=[],   # not needed after embedding
        ent_emb=ent_emb.astype(np.float32),
        rel_emb=rel_emb.astype(np.float32),
        model_name=model_name,
        device=device,
    )
