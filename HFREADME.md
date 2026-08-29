---
language:
- tl
- en
pretty_name: TagaSenti
license: cc-by-sa-4.0
task_categories:
- text-classification
tags:
- sentiment-analysis
- filipino
- tagalog
- taglish
- code-switching
- low-resource-language
- philippine-languages
- e-commerce
- news
- social-media
- adversarial
- llm-annotated
annotations_creators:
- machine-generated
- crowdsourced
- found
language_creators:
- crowdsourced
- machine-generated
- found
size_categories:
- 10K<n<100K
source_datasets:
- uhhlt/fires
- davidjm/sentitaglish
- amazon_polarity
- NewsPH
- tiktok_hate_speech
task_ids:
- multi-class-classification
---

# Dataset Card for TagaSenti

## Table of Contents
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)
  - [Contact](#contact)

## Dataset Description

- **Homepage:** [Hugging Face Datasets](https://huggingface.co/datasets/jjjardev/tagasenti)
- **Repository:** [GitHub](https://github.com/jjjardev/tagasenti)
- **Model:** [jjjardev/tagasenti_model](https://huggingface.co/jjjardev/tagasenti_model)
- **Paper:** forthcoming
- **Point of Contact:** Jessie James T. Jarder

### Dataset Summary

TagaSenti is a multi-domain sentiment-analysis dataset for **Tagalog and Taglish (Tagalog-English code-switching)**, spoken by over 80 million people in the Philippines. It contains **35,686** sentences sourced from e-commerce reviews, news articles, and social media — including **9,378 targeted adversarial sentences** across four development generations designed to strengthen model robustness on edge cases like sarcasm, negation, and ambiguous statements. Each sentence is annotated with a **Negative**, **Neutral**, or **Positive** sentiment label (mapped to integers 0, 1, and 2 respectively).

The dataset is composed of two layers:

**Layer 1 — Real-world base (25,000 rows):**

| Source | Domain | Language | Rows |
|--------|--------|----------|------|
| FiReCS (human gold) | E-commerce (Shopee, Lazada) | Tagalog/Taglish | 10,481 |
| SentiTaglish (human gold, collapsed 4→3 class) | Products & services | Tagalog/Taglish | 6,816 |
| Amazon Polarity (translated + synthetic neutral) | E-commerce (multi-category) | English→Tagalog | 3,942 |
| NewsPH (LLM-labeled) | News (politics, weather, crime, sports) | Tagalog | 5,999 |
| TikTok Hate Speech (LLM-labeled) | Social media (comments) | Tagalog+English | 6,292 |

**Layer 2 — Targeted adversarial augmentation (9,378 rows across 4 generations):**

| Generation | Version | Rows | Strategy |
|------------|---------|------|----------|
| 1st | v3 | 3,766 | 8 categories: sarcasm, negation, wish, indirect complaints, factual-neutral trigger, mixed/ambiguous, rhetorical Qs, hard positive |
| 2nd | v4 | 3,413 | "No keywords" — forbid sentiment keywords; forced-Negative to rebalance class skew |
| 3rd | v5/v5.1 | 2,417 | 8 new categories from v3 benchmark error analysis |
| 4th | v6 | 1,399 | Selective: Positive hard cases, Neutral hedging bias, Idiomatic expressions |

A companion model — fine-tuned `xlm-roberta-large` on this dataset — achieves **86.6% test accuracy** (macro‑F1 0.866). The v4 checkpoint is the production model, balancing Tagalog accuracy (86.6%) with cross-lingual transfer to Hiligaynon (62.4% accuracy, F1 0.624).

### ⚠️ Important Notice

This dataset includes publicly available social media comments and news articles. No PII scrubbing or anonymization has been applied. Users who require anonymized data should apply their own NER/scrubbing pipeline before use.

### Supported Tasks and Leaderboards

- **text-classification / sentiment-analysis**: classify a Tagalog or Taglish sentence as Negative, Neutral, or Positive.
  *Metrics*: accuracy, precision, recall, macro‑F1.
  *Suggested model*: `xlm-roberta-large`.
- **cross-lingual zero-shot / transfer-learning**: use the multi‑domain corpus to study cross‑lingual transfer into other Philippine (Austronesian) languages.

The dataset may also be used for **code‑switching detection** and **dialect-variation studies**, though those tasks are not officially benchmarked.

### Languages

The dataset is primarily in **Tagalog** (ISO 639‑3 `tgl`, also known as Filipino). Code‑switched segments appear in **English** (`en`). The news domain sentences are mostly monolingual Tagalog, while the e-commerce, social-media, and adversarial domains exhibit frequent **Taglish** code‑switching.

## Dataset Structure

### Data Instances

```csv
sentence,label
"Ang ganda ng quality ng tela, worth it ang price!",2
"Wala pa ring update ang order ko hanggang ngayon.",0
"Sa aking palagay, hindi naman ito gaanong importante.",1
"Ang bait mo naman, pinagbigyan mo ako sa wakas.",0  (adversarial — sarcasm)
```

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `sentence` | `string` | The Tagalog/Taglish text (cased, punctuation preserved). May include code-switched English. |
| `label` | `int` | Sentiment label: **0** = Negative, **1** = Neutral, **2** = Positive. |

Label mapping:
- `0` → Negative
- `1` → Neutral
- `2` → Positive

### Data Splits

> **Hub vs offline splits:** The Hugging Face Hub hosts `jjjardev/tagasenti` as a single `train` split (35,686 rows) for flexibility. The table below shows the **recommended reproducible stratified 80/10/10 split** used by `scripts/train/TagaSenti.py` (seed 42). In offline development (v1–v4), adversarial rows were held to train-only so val/test stayed comparable to the unaugmented baseline — to reproduce that, filter by `domain == "adversarial"` before splitting (see training script notes).

| Split | Sentences | Negative (0) | Neutral (1) | Positive (2) |
|-------|-----------|--------------|-------------|--------------|
| Train | 28,549 | 10,782 | 9,216 | 8,551 |
| Validation | 3,569 | 1,190 | 1,190 | 1,189 |
| Test | 3,569 | 1,190 | 1,190 | 1,189 |
| **Total** | **35,686** | **13,162** | **11,596** | **10,929** |

*Deduplicated (exact sentence match, keep first): ~34,945 rows (741 duplicates removed, 113 with conflicting labels). Training script deduplicates by default (`CFG["deduplicate"]=True`) before splitting to prevent train/val/test leakage. All splits use stratified random sampling (seed=42).*

**Companion model:** [`jjjardev/tagasenti_model`](https://huggingface.co/jjjardev/tagasenti_model) — fine-tuned `xlm-roberta-large` on the deduplicated full dataset, **86.6% test accuracy**, macro-F1 **0.866**. Cross-lingual zero-shot on Hiligaynon (HiliSenti): **62.4% accuracy**, F1 **0.624**.

### Domain Breakdown

| Domain | Rows | % of Total |
|--------|------|-----------|
| E-commerce | 12,709 | **35.6%** |
| Social media | 6,292 | **17.6%** |
| News | 5,999 | **16.8%** |
| Adversarial (4 generations) | 9,378 | **26.3%** |
| Synthetic edge-cases (v2) | 1,308 | **3.7%** |

## Dataset Creation

### Curation Rationale

Tagalog (Filipino) is an under-resourced language in NLP despite having over 80 million speakers. Existing sentiment resources are limited to single-domain corpora (primarily e-commerce) or are proprietary. TagaSenti was created to provide a **multi-domain, balanced, openly-licensed** sentiment corpus that enables the development of robust Tagalog sentiment models.

The dataset was designed around three principles:
1. **Domain diversity** — force models to generalize across e-commerce, news, and social media registers rather than overfitting to a single domain.
2. **Class balance** — eliminate the skewed distributions typical of real-world data to avoid majority-class bias in trained models.
3. **Targeted adversarial augmentation** — identify common failure modes (sarcasm, negation, ambiguous statements) and generate high-quality adversarial examples to harden the model.

### Source Data

#### Data Collection and Processing

**Phase 1 — Base Dataset Assembly**  
Existing publicly available Tagalog and Taglish datasets were surveyed on Hugging Face, GitHub, and Kaggle. From an initial pool of 10 datasets (227,328 rows), 5 were retained after deduplication, license compatibility checks, and domain coverage analysis. All datasets were converted to a unified `sentence,label` format and cleaned (21,663 rows removed). A stratified sample of 25,000 rows was drawn at exactly 33.3% per class.

**Phase 2 — Iterative Adversarial Generation (4 generations)**  
After training each baseline, failure cases were analyzed and **narrow adversarial categories** were designed to target specific blind spots. Sentences were generated using **Gemini 3.1 Pro** with strict linguistic guardrails:
- Word blacklists to prevent trivial keyword matching
- No explicit sentiment keywords permitted (v2+)
- Batch rotation across 8+ prompts for lexical variety
- Human spot-checks per batch

7 malformed rows were detected and fixed during v3 assembly. Total generated: 9,378 rows.

**Text Normalization** (applied to all splits):
- **NFKC Unicode normalization** — ligatures, homoglyphs, half-width characters
- **Laughter canonicalization** — `hahaha...` / `HAHAHA` → `hahaha`
- **Reduplication expansion** — `abot2` → `abot abot`, `abot-abot` → `abot abot`
- **Slang expansion** — 25 high-confidence patterns (`wla→wala`, `dko→di ko`, `kc→kasi`, `sya→siya`)
- **Character deduplication** — ≥3 consecutive identical chars → 2
- **Whitespace collapse**
- **No lowercasing** — XLM-RoBERTa-large is a cased model

#### Who are the source data producers?

- **FiReCS** — E-commerce reviewers on Shopee and Lazada (Philippines)
- **SentiTaglish** — Online reviewers of products and services
- **Amazon Polarity** — Amazon.com reviewers (original English); translated by Gemini 3.1 Pro
- **NewsPH** — Philippine news websites (politics, weather, crime, sports)
- **TikTok Hate Speech** — TikTok commenters (Tagalog + English code-switching)
- Original dataset creators: FiReCS (Lovely Rose et al.), SentiTaglish (Scheda et al.), Amazon Polarity (McAuley et al.)
- Adversarial sentences: generated by Gemini 3.1 Pro, reviewed by the author

### Annotations

#### Annotation process

Labels were assigned through three mechanisms:

1. **Gold human labels** (FiReCS, SentiTaglish) — inherited from the original datasets. SentiTaglish's 4-class schema was collapsed to 3-class by merging labels 3 and 4 into Positive.
2. **LLM-as-annotator** (NewsPH, TikTok) — Deepseek V4 Pro assigned a single-pass zero-shot label per sentence.
3. **Adversarial** — labels were assigned by the generation prompt (explicitly targeting a specific class) with human spot-checking.

#### Who are the annotators?

- Deepseek V4 Pro — LLM annotator for NewsPH (5,999 rows) and TikTok (6,292 rows)
- Gemini 3.1 Pro — translator and synthetic neutral generator for Amazon; full adversarial generation (9,378 rows across 4 generations)
- Jessie James T. Jarder — human review of adversarial batches

### Personal and Sensitive Information

The dataset includes publicly available social media comments (TikTok) and news articles with named individuals and locations. No PII scrubbing or anonymization has been applied. The adversarial sentences are artificially generated and contain no personal information.

## Considerations for Using the Data

### Social Impact of Dataset

This dataset enables NLP research for one of the most widely spoken yet under-resourced languages in Southeast Asia. It is intended to:
- lower the barrier for building sentiment‑aware applications serving Filipino speakers,
- spur further work on Philippine language technology, and
- serve as a case study for LLM-assisted dataset creation and adversarial augmentation in low-resource settings.

### Discussion of Biases

- **Domain skew** — 35.6% of the data is e-commerce; news (16.8%) and social media (17.6%) are underrepresented when adversarial rows are included.
- **LLM label bias** — Labels from Deepseek V4 Pro were not validated against human ground truth. Systematic annotation biases from the LLM's pretraining data are present.
- **Translation artifacts** — Amazon reviews were machine-translated and exhibit compressed syntactic patterns (e.g., high frequency of `pero`/`kaso` as discourse markers).
- **Adversarial generation bias** — Adversarial sentences were generated by an English-centric LLM and may contain subtle non-native artefacts.
- **Socioeconomic skew** — TikTok and SentiTaglish content reflects specific demographic and socioeconomic contexts.

### Other Known Limitations

- Only 3 base domains are covered (e-commerce, news, social media). Movie reviews, food/restaurant, education, beauty/cosmetics, and government services are absent.
- The dataset supports basic three-class sentiment classification only. No labels for sarcasm, frustration, irony, or aspect-based sentiment (though adversarial categories target sarcasm and indirectness).
- Code-switching patterns are concentrated in the social media and Amazon domains; the news domain is mostly monolingual Tagalog.
- LLM label quality is unvalidated — no inter-annotator agreement metric is available.
- The dataset uses a single training seed (42); no variance estimates are available for the reported metrics.

## Additional Information

### Dataset Curators

Jessie James T. Jarder

### Licensing Information

The dataset is released under the [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/) licence. You are free to share and adapt the material for any purpose (including commercial), provided you give appropriate credit and distribute your contributions under the same licence.

**Component licenses:**
- FiReCS — CC BY 4.0
- SentiTaglish — CC BY 4.0
- Amazon Polarity — Apache 2.0
- NewsPH — Public domain (Philippine news articles)
- TikTok Hate Speech — CC BY-SA 4.0
- Adversarial data — CC BY-SA 4.0 (original generation)

**Model weights** are released under Apache 2.0.

### Citation Information

**BibTeX:**
```bibtex
@misc{jessie_james_jarder_2026,
  author       = {Jessie James Jarder},
  title        = {tagasenti (Revision 3ebda33)},
  year         = 2026,
  url          = {https://huggingface.co/datasets/jjjardev/tagasenti},
  doi          = {10.57967/hf/9620},
  publisher    = {Hugging Face}
}
```

**APA:**
Jarder, J. J. (2026). *tagasenti* (Revision 3ebda33) [Data set]. Hugging Face. https://doi.org/10.57967/hf/9620

### Contributions

Dataset design, curation, adversarial augmentation, model training, and evaluation performed by Jessie James T. Jarder.

### Contact

For questions, open an issue on the [GitHub repository](https://github.com/jjjardev/tagasenti).

---

## Dataset Card Authors

Jessie James T. Jarder

## Dataset Card Contact

For questions, open a discussion on the **Hugging Face dataset repository**.
