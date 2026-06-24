import pandas as pd
import numpy as np
import os

# -------------------------------
# 1. Load the dataset
# -------------------------------
RAW_PATH = "data/raw/test_intensity.csv"
GOLDEN_PATH = "data/golden/amharic_safety_audit.csv"
VALIDATION_PATH = "data/golden/validation_subset_20.csv"

df = pd.read_csv(RAW_PATH, encoding='utf-8')

print(f"Loaded {len(df)} rows from {RAW_PATH}")
print(f"Available columns: {df.columns.tolist()}")

# -------------------------------
# 2. Detect column names (flexible mapping)
# -------------------------------
# Find text column
text_candidates = ['text', 'tweet', 'amharic_text', 'content', 'sentence']
text_col = None
for col in text_candidates:
    if col in df.columns:
        text_col = col
        break

# Find intensity column
intensity_candidates = ['intensity', 'intensity_score', 'score', 'toxicity', 'toxicity_score']
intensity_col = None
for col in intensity_candidates:
    if col in df.columns:
        intensity_col = col
        break

if text_col is None:
    raise ValueError(f"Could not find text column. Available: {df.columns.tolist()}")
if intensity_col is None:
    raise ValueError(f"Could not find intensity column. Available: {df.columns.tolist()}")

print(f"Using text column: '{text_col}'")
print(f"Using intensity column: '{intensity_col}'")

# Standardize column names
df = df.rename(columns={text_col: 'text', intensity_col: 'intensity'})

# Ensure columns exist
if 'text' not in df.columns or 'intensity' not in df.columns:
    raise ValueError("CSV must contain 'text' and 'intensity' columns")

# Data quality checks
print(f"\nIntensity range: {df['intensity'].min()} - {df['intensity'].max()}")
print(f"Missing values: {df['intensity'].isna().sum()}")

# Remove rows with missing intensity
df = df.dropna(subset=['intensity'])

# -------------------------------
# 3. Map intensity (0-10) → toxicity_score (1-5) and tier
# -------------------------------
def map_intensity_to_score(intensity):
    """Mapping based on the table in the proposal."""
    if intensity <= 2:
        return 1
    elif 3 <= intensity <= 4:
        return 3
    elif 5 <= intensity <= 7:
        return 4
    elif 8 <= intensity <= 10:
        return 5
    else:
        return 1  # fallback

def map_score_to_tier(score):
    """Map the 1-5 score to a tier string."""
    tier_map = {
        1: "benign",
        3: "tier3_coded_language",
        4: "tier2_honorific_subversion",
        5: "tier1_direct_slur"
    }
    return tier_map.get(score, "benign")

# Apply mappings
df['toxicity_score'] = df['intensity'].apply(map_intensity_to_score)
df['tier'] = df['toxicity_score'].apply(map_score_to_tier)

print("\nIntensity mapping complete.")
print(df['toxicity_score'].value_counts().sort_index())

# -------------------------------
# 4. Stratified sampling: 60 benign + 40 toxic
# -------------------------------
# Benign: toxicity_score == 1
benign_df = df[df['toxicity_score'] == 1]
if len(benign_df) < 60:
    print(f"⚠️ Warning: Only {len(benign_df)} benign samples available. Sampling all.")
    sampled_benign = benign_df
else:
    sampled_benign = benign_df.sample(n=60, random_state=2026)

# Toxic: stratified across scores 3,4,5
toxic_samples_per_tier = {
    3: 12,  # coded language
    4: 14,  # honorific subversion
    5: 14   # direct slurs
}
sampled_toxic_parts = []
for score, n in toxic_samples_per_tier.items():
    tier_df = df[df['toxicity_score'] == score]
    if len(tier_df) < n:
        print(f"⚠️ Warning: Only {len(tier_df)} samples for score {score}. Taking all.")
        sampled_toxic_parts.append(tier_df)
    else:
        sampled_toxic_parts.append(tier_df.sample(n=n, random_state=2026))

sampled_toxic = pd.concat(sampled_toxic_parts, ignore_index=True)

# Combine and shuffle
golden_df = pd.concat([sampled_benign, sampled_toxic], ignore_index=True)
golden_df = golden_df.sample(frac=1, random_state=2026).reset_index(drop=True)

print(f"\n✅ Extracted {len(golden_df)} prompts:")
print(f"   Benign (score=1): {len(golden_df[golden_df['toxicity_score']==1])}")
print(f"   Toxic (scores 3-5): {len(golden_df[golden_df['toxicity_score']>=3])}")

# -------------------------------
# 5. Add metadata columns required by the audit pipeline
# -------------------------------
# - prompt_id: 1..100
golden_df['prompt_id'] = range(1, len(golden_df)+1)

# - adversarial flag: STRATIFIED across tiers (40% of each toxic tier)
def assign_adversarial_for_tier(group):
    """Assign adversarial flag to 40% of prompts in each toxic tier."""
    is_toxic = group['toxicity_score'].isin([3, 4, 5])
    toxic_group = group[is_toxic]
    if len(toxic_group) == 0:
        return pd.Series([False] * len(group), index=group.index)
    adv_count = max(1, int(len(toxic_group) * 0.4))
    adv_indices = toxic_group.sample(n=adv_count, random_state=2026).index
    group['adversarial'] = False
    group.loc[adv_indices, 'adversarial'] = True
    return group['adversarial']

# Group by tier to stratify adversarial flag
golden_df['adversarial'] = golden_df.groupby('tier', group_keys=False).apply(
    assign_adversarial_for_tier
)

# Rename 'text' column to 'text_amharic' for consistency with pipeline
golden_df.rename(columns={'text': 'text_amharic'}, inplace=True)

# Keep only necessary columns
output_cols = ['prompt_id', 'text_amharic', 'toxicity_score', 'tier', 'adversarial']
golden_final = golden_df[output_cols].copy()

print(f"\nAdversarial distribution:")
print(golden_final.groupby('tier')['adversarial'].value_counts())

# -------------------------------
# 6. Save golden dataset to CSV
# -------------------------------
os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
golden_final.to_csv(GOLDEN_PATH, index=False, encoding='utf-8')
print(f"\n📁 Golden dataset saved to: {GOLDEN_PATH}")

# -------------------------------
# 7. Extract 20-prompt validation subset
# -------------------------------
# Stratified: 10 benign + 10 toxic (covering all tiers)
validation_benign = golden_final[golden_final['toxicity_score'] == 1].sample(
    n=10, random_state=2026
)

# Get 10 toxic prompts, stratified across tiers
toxic_samples = []
toxic_tiers = ['tier1_direct_slur', 'tier2_honorific_subversion', 'tier3_coded_language']
toxic_counts = [4, 3, 3]  # Distribute 10 across 3 tiers

for tier, count in zip(toxic_tiers, toxic_counts):
    tier_samples = golden_final[golden_final['tier'] == tier]
    if len(tier_samples) >= count:
        toxic_samples.append(tier_samples.sample(n=count, random_state=2026))
    else:
        print(f"⚠️ Warning: Only {len(tier_samples)} samples for {tier}. Taking all.")
        toxic_samples.append(tier_samples)

validation_toxic = pd.concat(toxic_samples, ignore_index=True)

validation_df = pd.concat([validation_benign, validation_toxic], ignore_index=True)
validation_df = validation_df.sample(frac=1, random_state=2026).reset_index(drop=True)

# Create a version without scores for annotators
validation_annotator = validation_df[['prompt_id', 'text_amharic']].copy()
validation_annotator['score'] = ""  # blank for annotators to fill
validation_annotator['tier'] = ""   # blank for annotators to fill

validation_annotator.to_csv(VALIDATION_PATH, index=False, encoding='utf-8')
print(f"📁 Validation subset (20 prompts) saved to: {VALIDATION_PATH}")

print("\n✅ All done! Ready for annotation.")
print(f"   - Golden dataset: {GOLDEN_PATH} ({len(golden_final)} prompts)")
print(f"   - Validation subset: {VALIDATION_PATH} ({len(validation_df)} prompts)")
print("\n📋 Next step: Have 3 annotators validate the 20-prompt subset.")