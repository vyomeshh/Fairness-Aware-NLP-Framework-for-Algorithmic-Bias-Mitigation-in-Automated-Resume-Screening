"""Central, recorded configuration (IPR-I Table 3.3: configurable parameters).

Values here are DEFAULTS for development/smoke testing only. Final values
(masking categories, k, group definition Z, seed) must be fixed and documented
once the real dataset is selected.
"""
from dataclasses import dataclass, field

SEED = 42

NEUTRAL_TOKENS = {
    "PERSON": "[PERSON]",
    "LOCATION": "[LOCATION]",
    "PRONOUN": "[PRONOUN]",
}

# Gendered pronouns only; singular "they/them" is gender-neutral and kept.
GENDERED_PRONOUNS = {"he", "him", "his", "himself", "she", "her", "hers", "herself"}

# ---- Gazetteers used ONLY by the offline rule-based fallback parser ----------
# (the spaCy backend uses the statistical NER model instead). Deliberately small
# and avoids ambiguous words (e.g. "Mark", "Will") to limit false positives.
FIRST_NAMES = {
    "Alex", "Priya", "Rahul", "Aisha", "Mohammed", "Neha", "Amit", "Sneha",
    "Rohan", "Anjali", "Vikram", "Pooja", "Arjun", "Kavya", "John", "Emily",
    "Michael", "Sarah", "David", "Jessica", "Daniel", "Olivia", "Karan",
    "Meera", "Imran", "Fatima", "Sanjay", "Divya", "Rajesh", "Sunita",
    "Teena", "Vyomesh",
}
SURNAMES = {
    "Sharma", "Verma", "Singh", "Khan", "Gupta", "Patel", "Kumar", "Mishra",
    "Panwar", "Smith", "Johnson", "Brown", "Miller", "Wilson", "Rao", "Iyer",
    "Das", "Reddy", "Ali",
}
LOCATIONS = {
    "Delhi", "New Delhi", "Mumbai", "Noida", "Greater Noida", "Gurugram",
    "Bengaluru", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune",
    "Lucknow", "Patna", "Jaipur", "Uttar Pradesh", "Bihar", "Kerala", "India",
    "London", "New York", "California", "Texas",
}


@dataclass(frozen=True)
class MaskConfig:
    categories: tuple = ("PERSON", "LOCATION", "PRONOUN")
    neutral_tokens: dict = field(default_factory=lambda: dict(NEUTRAL_TOKENS))
    seed: int = SEED
    top_k: int = 5            # shortlist size used to define Y_hat = 1
    context_depth: int = 1    # dependency-neighbourhood depth written to audit log
