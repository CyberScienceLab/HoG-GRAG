# Defense Against Knowledge Poisoning Attack on GraphRAG

## 🧠 Overview

This repository contains the code and experiments for a defense framework against knowledge poisoning attacks in GraphRAG-style multi-hop QA systems. **Auto-Immune GraphRAG** operates directly on hop-level execution traces over the knowledge graph and introduces a two-stage defense:

1. **Hop-wise poisoning detection**, which flags poisoned questions by identifying structural disruptions in the reasoning chain.
2. **Structural repair**, which edits the retrieved subgraph by removing suspect entities and relations and, when necessary, injecting minimal additional evidence from the global knowledge graph.

<img src="figures/overview2.jpg" />

We evaluate this framework on multi-hop question answering using:

- **Datasets**
  - [MuSiQue](https://github.com/StonyBrookNLP/musique)
  - [HotpotQA](https://github.com/hotpotqa/hotpot)

- **GraphRAG Pipelines**
  - [Microsoft GraphRAG](https://github.com/microsoft/graphrag)
  - [LightRAG](https://github.com/HKUDS/LightRAG)


### 📂 Repository Structure

This repository is organized around the two core components of Auto-Immune GraphRAG — detection and repair — with supporting modules for evaluation.

```text
AutoImmune-GraphRAG/
│
├── data/                         
│   ├── musique/                 
│   ├── hotpotqa/                 
├── prompts/ 
│   │   └── Context-Restricted Answering.md
│   │   └── Poison Text Generation.md
│   │   └── Question Paraphrasing.md
│   │   └── Response Evaluation.md
├── src/
│   ├── poison_text_generation.py
│   ├── GraphRAG_response.py     
│   ├── detection.py               
│   ├── repairer.py             
│   ├── repair_trace_analysis.py               
│   ├── evaluation.py
├── baselines/
│   │   └── Query_Paraphrasing.py
│   │   └── Perplexity_based.py
├── requirements.txt             
└── README.md
```

## 📄 Citation

If you use this methodology in your research, please cite:

> Havva Alizadeh Noughabi, Fattane Zarrinkalam, Ali Dehghantanha, **Defense Against Knowledge Poisoning Attack on GraphRAG**, Accepted at the Annual Meeting of the Association for Computational Linguistics (ACL 2026)  
