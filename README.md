
# LLaMA-Powered Climate Adaptation Evidence Synthesis

## Overview
This repository contains the code, data pipeline, and evaluation framework for our AI capstone project: **Structured Climate Adaptation Evidence Synthesis Using LLaMA**. The project explores whether fine-tuned open-source large language models (LLMs) like LLaMA 3.2B can replicate expert-level evidence annotation on a domain-specific dataset.

We demonstrate how retrieval-augmented generation (RAG), supervised fine-tuning (SFT), and prompt engineering can be combined to extract structured data (Geography, Stakeholders, Adaptation Depth) from scientific literature related to climate change adaptation.

## 🔍 Problem Statement
Manual evidence synthesis is a time-consuming process, especially in fast-growing domains like climate adaptation. We investigate whether LLaMA models, with retrieval and prompt design, can:
- Extract structured adaptation data from research papers.
- Reduce expert labeling time and cost.
- Match human-level accuracy on specific evidence fields.

## 📊 Dataset
- **Source**: GAMI (Global Adaptation Mapping Initiative) dataset (Berrang-Ford et al., 2021)
- **Subset**: Food security focus group (n = 551 articles)
- **Split**: Training (440), Test (111)
- **Features Used**:
  - Geographic Location (Low expertise)
  - Stakeholders (Intermediate expertise)
  - Depth of Adaptation (High expertise)
- **Format**: Markdown chunks with CSV annotations

## 🧱 Architecture & Components
1. **Chunking**: Simple word-based chunking (300 words per chunk)
2. **Embedding**: SentenceTransformers `all-MiniLM-L6-v2` used for semantic embeddings
   - Lightweight, efficient, 384-dim, ideal for fast FAISS indexing
3. **Indexing**: FAISS per-document index for top-k retrieval
4. **Prompting**: Multi-part structured prompts including:
   - Country
   - Stakeholders (taxonomy-based)
   - Depth + Explanation
5. **Fine-Tuning**:
   - **LoRA + Unsloth** implementation
   - 20 Epochs, bf16 precision, 4-bit quantized weights
   - Contexts: 5 chunks (train); 1,2,3,5,8 (test)
6. **Inference**: Temperature sweep (0.1 to 1.5)
7. **Evaluation**: Exact match + multi-label string overlap (stakeholders)

## 📁 Directory Structure
```
├── Accuracy_Images/
│   ├── Epochs_10_results/                # Contains accuracy_context_*.jpg
│   └── Epochs_20_results/                # Contains results.jpg (final evaluation)
├── Dataset_for_Finetunning/             # Contains fine-tuning dataset in .parquet format
├── Dataset_Pickle_Files/                # Raw GAMI markdown + human_annotations.csv
├── PPT/                                 # Project presentation slides
├── Test_Dataset_and_results/            # Test prompts + model outputs in .parquet format
├── code/                                # Two Jupyter notebooks for pipeline & fine-tuning
│   ├── *.ipynb
├── LICENSE
├── README.md
```

## 🧪 Training Strategy
### Phase 1: Dummy Fine-Tuning
- Purpose: Teach model the output format
- Accuracy: Low (Country 32%, Stakeholders 0%, Depth 13%)

### Phase 2: Real Dataset Fine-Tuning
- Input: 5 retrieved contexts
- Accuracy at Temperature 0.5:
  - Country: 88.3%
  - Stakeholders: 86.5%
  - Depth: 70.3%

### Heatmap/Lineplot Visualization
See `Accuracy_Images/` for temperature-context effect visualizations.

## 📊 Evaluation Metric
- **Country/Depth**: Exact match
- **Stakeholders**: Multi-value token overlap ("|||")

## 🛠️ Technologies Used
- LLaMA 3.2B via Unsloth (quantized)
- SentenceTransformers (MiniLM)
- FAISS for dense retrieval
- PyTorch + Hugging Face 
- Google Colab Pro (A100 GPU)

## 📈 Key Findings
- Structured prompting + fine-tuning boosts LLM reliability
- Depth remains the hardest task due to abstract, implicit reasoning
- Best performance at 5-contexts and temperature 0.5

## 📌 Future Work
- Extend to other sectors in GAMI (health, cities)
- Multi-document synthesis
- Integrate human-in-the-loop corrections
- Explore new open models (Mistral, Falcon)

## 📚 Citation
Berrang-Ford et al., 2021. A global stocktake of evidence on human adaptation to climate change. *Nature Climate Change*. https://doi.org/10.1038/s41558-021-01170-y

---
For questions, please contact:
**Sathwik Kuchana**, **Sri Sai Lalitha Mallika Yeturi**, **Priyadarshini Munigala**  
Yeshiva University, New York  
{skuchana, syeturi, pmunigal}@mail.yu.edu
