# src/risk_model/taxonomy.py
"""Demographic attribute taxonomy and integer identifier mappings.

Defines coarse classification categories for age, education, occupation, and location,
ensuring balanced class support and consistent integer encoding across risk models.

Two reconciliation decisions are recorded here (see docs/methodology-review.md
dated note 2026-11, decisions #4 and #7):

- EDUCATION adopts the SynthPAI paper's four-way scale (high_school / college /
  master / phd) for comparability with the paper. It replaces the previous ad
  hoc split (high_school / undergraduate / graduate_professional / other_unknown).
- LOCATION keeps the coarse regional buckets ONLY as an auxiliary/diagnostic
  signal for the multi-task heads. The reportable location metric is the
  free-text generate-and-score path in src/evaluation/attacker.py
  (Phi4MiniAttacker.evaluate_success), which scores a guessed free-text
  city/state/region token-wise against the ground truth. The Europe-vs-Other
  style collapse makes any coarse location accuracy number near-meaningless, so
  it must not be reported as the headline location result.
"""

from typing import Dict, List
from src.contracts.schemas import AttributeName

ATTRIBUTE_TAXONOMY: Dict[AttributeName, List[str]] = {
    AttributeName.AGE: [
        "18-24",
        "25-34",
        "35-44",
        "45-54",
        "55+",
    ],
    # SynthPAI paper's four-way education scale (high school / college / master /
    # PhD; paper support college 38.3% / master 37.7% / PhD 15.7% / high school
    # 8.3%). Canonical snake_case keys keep comparability with the paper while
    # matching the project's label-key style. The paper scale has no "other"
    # bucket, so map_to_taxonomy_id in train.py defaults unmatched text to
    # "high_school" (the least-attainment, lowest-information class). See
    # docs/methodology-review.md dated note (2026-11) for the rationale.
    AttributeName.EDUCATION: [
        "high_school",
        "college",
        "master",
        "phd",
    ],
    AttributeName.OCCUPATION: [
        "student",
        "technology",
        "healthcare",
        "education_academic",
        "business_finance",
        "service_retail",
        "other",
    ],
    AttributeName.LOCATION: [
        "us_northeast",
        "us_midwest",
        "us_south",
        "us_west",
        "uk_europe",
        "canada",
        "asia_pacific",
        "other_global",
    ],
}

LABEL_TO_ID: Dict[AttributeName, Dict[str, int]] = {
    attr: {label: idx for idx, label in enumerate(labels)}
    for attr, labels in ATTRIBUTE_TAXONOMY.items()
}

ID_TO_LABEL: Dict[AttributeName, Dict[int, str]] = {
    attr: {idx: label for idx, label in enumerate(labels)}
    for attr, labels in ATTRIBUTE_TAXONOMY.items()
}
