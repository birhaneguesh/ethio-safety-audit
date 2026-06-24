# Ethio‑Safety Audit

**Localized Amharic LLM Safety Evaluation Toolkit**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## Overview

**Ethio‑Safety Audit** is an open‑source toolkit that tests whether AI models can detect Amharic hate speech. Current safety systems, trained mostly on English, miss culturally specific abuse—like insults hidden behind formal language or historical references—risking AI‑amplified ethnic conflict in Ethiopia.

Our audit of 100 validated prompts found that the best open‑weight judge detects only **44.7%** of toxic Amharic content, missing over half; yet it flags **zero** benign prompts as toxic (0% false positives). Novelty lies in our **three‑tier Amharic harm taxonomy** (direct slurs, honorific subversion, coded language) developed with native speakers, and a transparent pipeline that evaluates prompts directly—avoiding the hallucination confounds of response‑based scoring.

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


