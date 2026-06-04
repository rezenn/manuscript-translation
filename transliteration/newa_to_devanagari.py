"""
newa_to_devanagari.py  —  Newa Class Names → Devanagari (v2)
══════════════════════════════════════════════════════════════

BUG FIXED vs v1
───────────────
v1 had CHAR_MAP keys as lowercase ("vowel_a") but the class_map.json
stores them as mixed case ("vowel_A", "vowel_AA"). The ALIASES table
tried to patch this but predictions_to_text() didn't always go through
get_char_info(). Now ALL lookups go through get_char_info() which does:
  1. Exact match
  2. Lowercase match
  3. Alias match (handles all known alternates)
  4. Returns ⟨?⟩ if nothing found (never crashes)
"""

# ── PRIMARY CHARACTER TABLE ──────────────────────────────────────
# Keys: the class names used in your class_map.json / model output.
# Values: deva (Devanagari), iast (romanization), desc (human label).

CHAR_MAP = {
    # Consonants (33)
    "ka":   {"deva": "क",  "iast": "ka"},
    "kha":  {"deva": "ख",  "iast": "kha"},
    "ga":   {"deva": "ग",  "iast": "ga"},
    "gha":  {"deva": "घ",  "iast": "gha"},
    "nga":  {"deva": "ङ",  "iast": "ṅa"},
    "ca":   {"deva": "च",  "iast": "ca"},
    "cha":  {"deva": "छ",  "iast": "cha"},
    "ja":   {"deva": "ज",  "iast": "ja"},
    "jha":  {"deva": "झ",  "iast": "jha"},
    "nya":  {"deva": "ञ",  "iast": "ña"},
    "tta":  {"deva": "ट",  "iast": "ṭa"},
    "ttha": {"deva": "ठ",  "iast": "ṭha"},
    "dda":  {"deva": "ड",  "iast": "ḍa"},
    "ddha": {"deva": "ढ",  "iast": "ḍha"},
    "nna":  {"deva": "ण",  "iast": "ṇa"},
    "ta":   {"deva": "त",  "iast": "ta"},
    "tha":  {"deva": "थ",  "iast": "tha"},
    "da":   {"deva": "द",  "iast": "da"},
    "dha":  {"deva": "ध",  "iast": "dha"},
    "na":   {"deva": "न",  "iast": "na"},
    "pa":   {"deva": "प",  "iast": "pa"},
    "pha":  {"deva": "फ",  "iast": "pha"},
    "ba":   {"deva": "ब",  "iast": "ba"},
    "bha":  {"deva": "भ",  "iast": "bha"},
    "ma":   {"deva": "म",  "iast": "ma"},
    "ya":   {"deva": "य",  "iast": "ya"},
    "ra":   {"deva": "र",  "iast": "ra"},
    "la":   {"deva": "ल",  "iast": "la"},
    "wa":   {"deva": "व",  "iast": "va"},
    "sha":  {"deva": "श",  "iast": "śa"},
    "ssa":  {"deva": "ष",  "iast": "ṣa"},
    "sa":   {"deva": "स",  "iast": "sa"},
    "ha":   {"deva": "ह",  "iast": "ha"},

    # Dependent vowel signs / matras (9)
    "matra_aa": {"deva": "ा",  "iast": "ā"},
    "matra_i":  {"deva": "ि",  "iast": "i"},
    "matra_ii": {"deva": "ी",  "iast": "ī"},
    "matra_u":  {"deva": "ु",  "iast": "u"},
    "matra_uu": {"deva": "ू",  "iast": "ū"},
    "matra_e":  {"deva": "े",  "iast": "e"},
    "matra_ai": {"deva": "ै",  "iast": "ai"},
    "matra_o":  {"deva": "ो",  "iast": "o"},
    "matra_au": {"deva": "ौ",  "iast": "au"},

    # Independent vowels (10)
    # All stored in lowercase; uppercase variants handled via get_char_info
    "vowel_a":   {"deva": "अ",  "iast": "a"},
    "vowel_aa":  {"deva": "आ",  "iast": "ā"},
    "vowel_i":   {"deva": "इ",  "iast": "i"},
    "vowel_ii":  {"deva": "ई",  "iast": "ī"},
    "vowel_u":   {"deva": "उ",  "iast": "u"},
    "vowel_uu":  {"deva": "ऊ",  "iast": "ū"},
    "vowel_e":   {"deva": "ए",  "iast": "e"},
    "vowel_ai":  {"deva": "ऐ",  "iast": "ai"},
    "vowel_o":   {"deva": "ओ",  "iast": "o"},
    "vowel_au":  {"deva": "औ",  "iast": "au"},

    # Special signs (5)
    "anusvara":    {"deva": "ं",  "iast": "ṃ"},
    "candrabindu": {"deva": "ँ",  "iast": "m̐"},
    "visarga":     {"deva": "ः",  "iast": "ḥ"},
    "virama":      {"deva": "्",  "iast": ""},
    "avagraha":    {"deva": "ऽ",  "iast": "ʼ"},

    # Digits (10)
    "digit_0": {"deva": "०",  "iast": "0"},
    "digit_1": {"deva": "१",  "iast": "1"},
    "digit_2": {"deva": "२",  "iast": "2"},
    "digit_3": {"deva": "३",  "iast": "3"},
    "digit_4": {"deva": "४",  "iast": "4"},
    "digit_5": {"deva": "५",  "iast": "5"},
    "digit_6": {"deva": "६",  "iast": "6"},
    "digit_7": {"deva": "७",  "iast": "7"},
    "digit_8": {"deva": "८",  "iast": "8"},
    "digit_9": {"deva": "९",  "iast": "9"},

    # Punctuation
    "danda":        {"deva": "।",  "iast": "."},
    "double_danda": {"deva": "॥",  "iast": ".."},
    "space":        {"deva": " ",  "iast": " "},
}

# ── ALIASES ──────────────────────────────────────────────────────
# Handles all known alternate spellings / capitalisations.
# Maps alternate → canonical CHAR_MAP key.

ALIASES: dict = {}

# Auto-generate vowel_X (uppercase) → vowel_x (lowercase)
for _v in ["a", "aa", "i", "ii", "u", "uu", "e", "ai", "o", "au"]:
    ALIASES[f"vowel_{_v.upper()}"] = f"vowel_{_v}"
    ALIASES[f"vowel_{_v}"]         = f"vowel_{_v}"   # idempotent

# Legacy matra names
for _src, _dst in [
    ("aa_sign",  "matra_aa"),
    ("i_sign",   "matra_i"),
    ("ii_sign",  "matra_ii"),
    ("u_sign",   "matra_u"),
    ("uu_sign",  "matra_uu"),
    ("e_sign",   "matra_e"),
    ("ai_sign",  "matra_ai"),
    ("o_sign",   "matra_o"),
    ("au_sign",  "matra_au"),
]:
    ALIASES[_src] = _dst

# Other common alternates
ALIASES.update({
    "halant":       "virama",
    "chandrabindu": "candrabindu",
    "anuswar":      "anusvara",
    "visarg":       "visarga",
})


# ══════════════════════════════════════════════════════════════════
# LOOKUP FUNCTION
# ══════════════════════════════════════════════════════════════════

def get_char_info(class_name: str) -> dict:
    """
    Look up a class name and return its Devanagari/IAST info.

    Resolution order:
      1. Exact match in CHAR_MAP
      2. Lowercase match in CHAR_MAP
      3. Alias → canonical → CHAR_MAP
      4. Lowercase alias
      5. Return placeholder  {"deva": "⟨?⟩", "iast": "[name]"}

    Never raises — always returns a dict.
    """
    name  = class_name.strip()
    lower = name.lower()

    # 1. Exact
    if name in CHAR_MAP:
        return CHAR_MAP[name]

    # 2. Lowercase
    if lower in CHAR_MAP:
        return CHAR_MAP[lower]

    # 3. Alias (original case)
    if name in ALIASES:
        target = ALIASES[name]
        if target in CHAR_MAP:
            return CHAR_MAP[target]

    # 4. Alias (lowercase)
    if lower in ALIASES:
        target = ALIASES[lower]
        if target in CHAR_MAP:
            return CHAR_MAP[target]

    # 5. Not found
    return {"deva": "⟨?⟩", "iast": f"[{name}]"}


# ══════════════════════════════════════════════════════════════════
# CONVERSION HELPERS
# ══════════════════════════════════════════════════════════════════

def convert_sequence(class_names: list, output_format: str = "devanagari") -> str:
    """Convert a list of class names to a Devanagari or IAST string."""
    out = []
    for name in class_names:
        info = get_char_info(name)
        out.append(info["iast"] if output_format == "iast" else info["deva"])
    return "".join(out)


def predictions_to_text(
    predictions: list,
    output_format: str = "devanagari",
    low_conf_marker: str = "⟨?⟩",
) -> dict:
    """
    Convert a list of prediction dicts (from recognize.py) into text.

    Each prediction dict must have at least:
      "line"      (int)
      "predicted" (str class name)
      "low_conf"  (bool)
      "confidence" (float)

    Returns:
    {
        "text":    full text (lines joined by newline),
        "lines":   list of per-line strings,
        "by_line": {line_idx: string},
        "flagged": list of low-confidence predictions,
    }
    """
    lines_dict: dict = {}
    flagged:    list = []

    for pred in predictions:
        line_idx   = pred.get("line", 0)
        class_name = pred.get("predicted") or "space"
        is_low     = pred.get("low_conf", False)
        confidence = pred.get("confidence", 1.0)

        if is_low:
            char = low_conf_marker
            flagged.append({
                "file":       pred.get("file", ""),
                "line":       line_idx,
                "char_idx":   pred.get("char_idx", 0),
                "predicted":  class_name,
                "confidence": confidence,
                "top5":       pred.get("top5", []),
            })
        else:
            info = get_char_info(class_name)
            char = info["iast"] if output_format == "iast" else info["deva"]

        lines_dict.setdefault(line_idx, []).append(char)

    by_line   = {idx: "".join(chars) for idx, chars in sorted(lines_dict.items())}
    full_text = "\n".join(by_line.values())

    return {
        "text":    full_text,
        "lines":   list(by_line.values()),
        "by_line": by_line,
        "flagged": flagged,
    }


# ══════════════════════════════════════════════════════════════════
# COVERAGE CHECK (for development)
# ══════════════════════════════════════════════════════════════════

def check_coverage(class_list: list) -> None:
    print(f"\nChecking coverage for {len(class_list)} classes...\n")
    unmapped = []
    for name in sorted(class_list):
        info = get_char_info(name)
        ok   = "⟨?⟩" not in info["deva"]
        mark = "✓" if ok else "✗ UNMAPPED"
        if not ok:
            unmapped.append(name)
        print(f"  {mark:12s} {name:25s} → {info['deva']}  ({info['iast']})")

    print(f"\n{'─'*55}")
    if unmapped:
        print(f"⚠  {len(unmapped)} unmapped: {unmapped}")
    else:
        print(f"✓ All {len(class_list)} classes mapped!")


# ══════════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import pathlib, json

    cm_path = pathlib.Path("dataset_final/class_map.json")
    if cm_path.exists():
        with open(cm_path, encoding="utf-8") as f:
            cm = json.load(f)
        classes = sorted(cm.keys())
        print(f"Loaded {len(classes)} classes from {cm_path}")
    else:
        classes = [
            "anusvara", "avagraha", "ba", "bha", "ca", "candrabindu",
            "cha", "da", "dda", "ddha", "dha",
            "digit_0","digit_1","digit_2","digit_3","digit_4",
            "digit_5","digit_6","digit_7","digit_8","digit_9",
            "ga", "gha", "ha", "ja", "jha", "ka", "kha", "la", "ma",
            "matra_aa","matra_ai","matra_au","matra_e","matra_i",
            "matra_ii","matra_o","matra_u","matra_uu",
            "na", "nga", "nna", "nya", "pa", "pha", "ra", "sa", "sha",
            "ssa", "ta", "tha", "tta", "ttha", "virama", "visarga",
            "vowel_A","vowel_AA","vowel_AI","vowel_AU","vowel_E",
            "vowel_I","vowel_II","vowel_O","vowel_U","vowel_UU",
            "wa", "ya",
        ]

    check_coverage(classes)

    print("\n── Sequence test ──")
    seq = ["ka", "matra_aa", "la", "virama", "pa", "matra_aa", "danda"]
    print(f"Input:      {seq}")
    print(f"Devanagari: {convert_sequence(seq)}")
    print(f"IAST:       {convert_sequence(seq, 'iast')}")

    print("\n── Uppercase vowel test (the v1 bug) ──")
    seq2 = ["vowel_A", "vowel_AA", "vowel_I", "vowel_UU", "vowel_O"]
    print(f"Input:      {seq2}")
    print(f"Devanagari: {convert_sequence(seq2)}")
    assert "⟨?⟩" not in convert_sequence(seq2), "BUG: uppercase vowels not resolved!"
    print("✓ No unmapped characters")