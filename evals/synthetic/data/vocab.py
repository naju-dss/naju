"""Shared synthetic vocabulary used by both tasks and all models.

Token layout (integer ids):
  0..NUM_SPECIAL-1            : structural / special tokens
  ENTITY_BASE..+NUM_ENTITIES  : entity-name tokens (pool of distinct entities)
  VALUE_BASE ..+NUM_VALUES    : value / state tokens (also the label space)

The classification label is the *value index* in [0, NUM_VALUES).
"""

# --- special tokens ---
PAD = 0
BOS = 1
SEP = 2          # sentence separator ("." )
KEY = 3          # "the code of"
IS = 4           # "is"
QUES = 5         # "question:"
NOW = 6          # "...now?"  (state tracking query)
INIT = 7         # "initially / at the beginning"
LATER = 8        # "later / after"
FINAL = 9        # "finally"
LIKES = 10       # "prefers / likes"
MOVED = 11       # generic distractor predicate
EOS = 12
NUM_SPECIAL = 16

# --- entity & value pools ---
ENTITY_BASE = NUM_SPECIAL
NUM_ENTITIES = 32
VALUE_BASE = ENTITY_BASE + NUM_ENTITIES
NUM_VALUES = 16

VOCAB_SIZE = VALUE_BASE + NUM_VALUES
NUM_CLASSES = NUM_VALUES


def entity_token(i: int) -> int:
    assert 0 <= i < NUM_ENTITIES
    return ENTITY_BASE + i


def value_token(v: int) -> int:
    assert 0 <= v < NUM_VALUES
    return VALUE_BASE + v
