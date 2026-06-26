# Ethio‑Safety Audit

**Localized Amharic LLM Safety Evaluation Toolkit**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status: Production](https://img.shields.io/badge/Status-Production-brightgreen.svg)]()

---

## Overview

**Ethio‑Safety Audit** is an open‑source toolkit that tests whether AI models can detect Amharic hate speech. Current safety systems, trained mostly on English, miss culturally specific abuse—like insults hidden behind formal language or historical references—risking AI‑amplified ethnic conflict in Ethiopia.

Our audit of **100 validated prompts** (99 valid evaluations) found that the best open‑weight judge (Qwen 72B) detects only **43.6%** of toxic Amharic content, missing over half; yet it flags **zero** benign prompts as toxic (0% false positives). This is the first quantified baseline for Amharic LLM safety.

### Methodological Contribution

Unlike existing safety benchmarks that score model responses (which can hallucinate safe text from toxic prompts), Ethio‑Safety Audit evaluates the **original Amharic prompt directly**. This eliminates the hallucination confound and provides a cleaner measure of whether a model's guardrails would detect a toxic input before generating a response.

---

## Key Results

| Metric | Value |
| :--- | :--- |
| **Detection Accuracy** | **43.6%** |
| **False Negative Rate** | **56.4%** |
| **Benign False Positive Rate** | **0%** |
| **Adversarial FNR** | 53.8% |
| **Valid Evaluations** | 99/100 |

**Judge Model:** Qwen/Qwen2.5-72B-Instruct (primary) with meta-llama/Llama-3.3-70B-Instruct as fallback.

**Ground Truth:** TRAC‑2024 dataset annotated by 5 native Amharic speakers.

### Why This Matters

The **0% false positive rate** is critical for real‑world deployment: moderators can trust that flagged content is genuinely toxic, avoiding the reputational harm of falsely censoring benign speech.

### Per‑Tier Detection Accuracy

| Tier | Accuracy |
| :--- | :--- |
| Tier 1 (Direct Slurs) | 50.0% |
| Tier 2 (Honorific Subversion) | 38.5% |
| Tier 3 (Coded Language) | 41.7% |

**Finding:** Tier 2 and Tier 3 detection is significantly lower, confirming that culturally subtle toxicity patterns are systematically missed by current models.

---

## Quickstart (5 Minutes)

```bash
# 1. Clone
git clone https://github.com/your-org/ethio-safety-audit.git
cd ethio-safety-audit

# 2. Set up environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure Hugging Face token
echo "HF_TOKEN=hf_your_token_here" > .env   # Get token from huggingface.co/settings/tokens

# 4. Run test (5 prompts)
python code/audit/run_judge_pipeline.py --test_limit 5