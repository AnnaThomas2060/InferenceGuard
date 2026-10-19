# src/risk_model/taxonomy.py
"""Demographic attribute taxonomy and integer identifier mappings.

Defines coarse classification categories for age, education, occupation, and location,
ensuring balanced class support and consistent integer encoding across risk models.
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
    AttributeName.EDUCATION: [
        "high_school",
        "undergraduate",
        "graduate_professional",
        "other_unknown",
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
