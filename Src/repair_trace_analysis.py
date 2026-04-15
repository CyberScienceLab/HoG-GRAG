import pandas as pd
import json

def parse_trace(trace_str: str):
    if pd.isna(trace_str):
        return []
    s = str(trace_str).strip()
    s = s.replace('""', '"').strip()
    # Ensure valid JSON list format
    if not s.startswith("["):
        s = "[" + s
    if not s.endswith("]"):
        s = s + "]"
    try:
        events = json.loads(s)
        return events if isinstance(events, list) else [events]
    except Exception:
        return []


def count_trace_events(csv_path: str, trace_col: str = "trace_json"):
    df = pd.read_csv(csv_path)
    event_counts = {}
    for trace_str in df[trace_col]:
        events = parse_trace(trace_str)
        for ev in events:
            etype = ev.get("event")
            if etype:
                event_counts[etype] = event_counts.get(etype, 0) + 1
    print("\n=== Event Counts Across All Traces ===")
    for etype, cnt in sorted(event_counts.items(), key=lambda x: -x[1]):
        print(f"{etype:28s} : {cnt}")
    return event_counts
