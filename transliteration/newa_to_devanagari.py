"""
newa_to_devanagari.py  —  Newa Class Names → Devanagari
════════════════════════════════════════════════════════

Maps EVERY class name from your actual class_map.json → Devanagari.

YOUR 67 CLASSES (from class_map.json):
  Consonants:  ka kha ga gha nga ca cha ja jha nya tta ttha dda ddha
               nna ta tha da dha na pa pha ba bha ma ya ra la wa
               sha ssa sa ha
  Matras:      matra_aa matra_i matra_ii matra_u matra_uu
               matra_e matra_ai matra_o matra_au
  Vowels:      vowel_A vowel_AA vowel_I vowel_II vowel_U vowel_UU
               vowel_E vowel_AI vowel_O vowel_AU
  Signs:       anusvara candrabindu visarga virama avagraha
  Digits:      digit_0 … digit_9

UNICODE BLOCK: U+11400 (Newa) — decimal 70656+
  70656 = U+11400 = 𑐀 (Newa Letter A)
  All class names map to Devanagari equivalents for readability.
"""

CHAR_MAP = {

    # ── CONSONANTS (33) ────────────────────────────────────────────
    "ka":   {"deva": "क",  "iast": "ka",    "desc": "ka"},
    "kha":  {"deva": "ख",  "iast": "kha",   "desc": "kha"},
    "ga":   {"deva": "ग",  "iast": "ga",    "desc": "ga"},
    "gha":  {"deva": "घ",  "iast": "gha",   "desc": "gha"},
    "nga":  {"deva": "ङ",  "iast": "ṅa",    "desc": "nga"},
    "ca":   {"deva": "च",  "iast": "ca",    "desc": "ca"},
    "cha":  {"deva": "छ",  "iast": "cha",   "desc": "cha"},
    "ja":   {"deva": "ज",  "iast": "ja",    "desc": "ja"},
    "jha":  {"deva": "झ",  "iast": "jha",   "desc": "jha"},
    "nya":  {"deva": "ञ",  "iast": "ña",    "desc": "nya"},
    "tta":  {"deva": "ट",  "iast": "ṭa",    "desc": "tta"},
    "ttha": {"deva": "ठ",  "iast": "ṭha",   "desc": "ttha"},
    "dda":  {"deva": "ड",  "iast": "ḍa",    "desc": "dda"},
    "ddha": {"deva": "ढ",  "iast": "ḍha",   "desc": "ddha"},
    "nna":  {"deva": "ण",  "iast": "ṇa",    "desc": "nna"},
    "ta":   {"deva": "त",  "iast": "ta",    "desc": "ta"},
    "tha":  {"deva": "थ",  "iast": "tha",   "desc": "tha"},
    "da":   {"deva": "द",  "iast": "da",    "desc": "da"},
    "dha":  {"deva": "ध",  "iast": "dha",   "desc": "dha"},
    "na":   {"deva": "न",  "iast": "na",    "desc": "na"},
    "pa":   {"deva": "प",  "iast": "pa",    "desc": "pa"},
    "pha":  {"deva": "फ",  "iast": "pha",   "desc": "pha"},
    "ba":   {"deva": "ब",  "iast": "ba",    "desc": "ba"},
    "bha":  {"deva": "भ",  "iast": "bha",   "desc": "bha"},
    "ma":   {"deva": "म",  "iast": "ma",    "desc": "ma"},
    "ya":   {"deva": "य",  "iast": "ya",    "desc": "ya"},
    "ra":   {"deva": "र",  "iast": "ra",    "desc": "ra"},
    "la":   {"deva": "ल",  "iast": "la",    "desc": "la"},
    "wa":   {"deva": "व",  "iast": "va",    "desc": "wa/va"},  # ← wa maps to व
    "sha":  {"deva": "श",  "iast": "śa",    "desc": "sha"},
    "ssa":  {"deva": "ष",  "iast": "ṣa",    "desc": "ssa"},
    "sa":   {"deva": "स",  "iast": "sa",    "desc": "sa"},
    "ha":   {"deva": "ह",  "iast": "ha",    "desc": "ha"},

    # ── DEPENDENT VOWEL SIGNS / MATRAS (9) ─────────────────────────
    # These attach to a preceding consonant.
    # e.g.  ka + matra_aa → का (kā)
    #        ka + matra_i  → कि (ki)
    "matra_aa": {"deva": "ा",  "iast": "ā",   "desc": "aa matra (𑐵)"},
    "matra_i":  {"deva": "ि",  "iast": "i",   "desc": "i matra (𑐶)"},
    "matra_ii": {"deva": "ी",  "iast": "ī",   "desc": "ii matra (𑐷)"},
    "matra_u":  {"deva": "ु",  "iast": "u",   "desc": "u matra (𑐸)"},
    "matra_uu": {"deva": "ू",  "iast": "ū",   "desc": "uu matra (𑐹)"},
    "matra_e":  {"deva": "े",  "iast": "e",   "desc": "e matra (𑐾)"},
    "matra_ai": {"deva": "ै",  "iast": "ai",  "desc": "ai matra (𑐿)"},
    "matra_o":  {"deva": "ो",  "iast": "o",   "desc": "o matra (𑑀)"},
    "matra_au": {"deva": "ौ",  "iast": "au",  "desc": "au matra (𑑁)"},

    # ── INDEPENDENT VOWELS (10) ────────────────────────────────────
    # Used at the START of a word (not after a consonant).
    # e.g. vowel_A = अ (standalone 'a'), vowel_AA = आ (standalone 'aa')
    "vowel_a":   {"deva": "अ",  "iast": "a",   "desc": "vowel A (𑐀)"},
    "vowel_aa":  {"deva": "आ",  "iast": "ā",   "desc": "vowel AA (𑐁)"},
    "vowel_i":   {"deva": "इ",  "iast": "i",   "desc": "vowel I (𑐂)"},
    "vowel_ii":  {"deva": "ई",  "iast": "ī",   "desc": "vowel II (𑐃)"},
    "vowel_u":   {"deva": "उ",  "iast": "u",   "desc": "vowel U (𑐄)"},
    "vowel_uu":  {"deva": "ऊ",  "iast": "ū",   "desc": "vowel UU (𑐅)"},
    "vowel_e":   {"deva": "ए",  "iast": "e",   "desc": "vowel E (𑐊)"},
    "vowel_ai":  {"deva": "ऐ",  "iast": "ai",  "desc": "vowel AI (𑐋)"},
    "vowel_o":   {"deva": "ओ",  "iast": "o",   "desc": "vowel O (𑐌)"},
    "vowel_au":  {"deva": "औ",  "iast": "au",  "desc": "vowel AU (𑐍)"},

    # ── SPECIAL SIGNS (5) ──────────────────────────────────────────
    "anusvara":    {"deva": "ं",  "iast": "ṃ",   "desc": "anusvara/sinhaphuti (𑑄)"},
    "candrabindu": {"deva": "ँ",  "iast": "m̐",   "desc": "candrabindu/milaaphuti (𑑃)"},
    "visarga":     {"deva": "ः",  "iast": "ḥ",   "desc": "visarga/liphuti (𑑅)"},
    "virama":      {"deva": "्",  "iast": "",    "desc": "virama/tutisaalaa (𑑂) — removes inherent a"},
    "avagraha":    {"deva": "ऽ",  "iast": "ʼ",   "desc": "avagraha/sulaa (𑑇)"},

    # ── DIGITS (10) ────────────────────────────────────────────────
    "digit_0":  {"deva": "०",  "iast": "0",  "desc": "digit 0 (𑑐)"},
    "digit_1":  {"deva": "१",  "iast": "1",  "desc": "digit 1 (𑑑)"},
    "digit_2":  {"deva": "२",  "iast": "2",  "desc": "digit 2 (𑑒)"},
    "digit_3":  {"deva": "३",  "iast": "3",  "desc": "digit 3 (𑑓)"},
    "digit_4":  {"deva": "४",  "iast": "4",  "desc": "digit 4 (𑑔)"},
    "digit_5":  {"deva": "५",  "iast": "5",  "desc": "digit 5 (𑑕)"},
    "digit_6":  {"deva": "६",  "iast": "6",  "desc": "digit 6 (𑑖)"},
    "digit_7":  {"deva": "७",  "iast": "7",  "desc": "digit 7 (𑑗)"},
    "digit_8":  {"deva": "८",  "iast": "8",  "desc": "digit 8 (𑑘)"},
    "digit_9":  {"deva": "९",  "iast": "9",  "desc": "digit 9 (𑑙)"},

    # ── PUNCTUATION ────────────────────────────────────────────────
    "danda":        {"deva": "।",  "iast": ".",   "desc": "danda (sentence end)"},
    "double_danda": {"deva": "॥",  "iast": "..",  "desc": "double danda (section end)"},
    "space":        {"deva": " ",  "iast": " ",   "desc": "space"},
}

# ── Aliases ───────────────────────────────────────────────────────
# Maps alternate spellings → canonical key in CHAR_MAP above.
# Your class_map.json uses mixed case (vowel_A, vowel_AA) — handle both.
ALIASES = {
    # vowel_X (uppercase) → vowel_x (lowercase key in CHAR_MAP)
    "vowel_a":   "vowel_a",    # already lowercase
    "vowel_A":   "vowel_a",
    "vowel_aa":  "vowel_aa",
    "vowel_AA":  "vowel_aa",
    "vowel_i":   "vowel_i",
    "vowel_I":   "vowel_i",
    "vowel_ii":  "vowel_ii",
    "vowel_II":  "vowel_ii",
    "vowel_u":   "vowel_u",
    "vowel_U":   "vowel_u",
    "vowel_uu":  "vowel_uu",
    "vowel_UU":  "vowel_uu",
    "vowel_e":   "vowel_e",
    "vowel_E":   "vowel_e",
    "vowel_ai":  "vowel_ai",
    "vowel_AI":  "vowel_ai",
    "vowel_o":   "vowel_o",
    "vowel_O":   "vowel_o",
    "vowel_au":  "vowel_au",
    "vowel_AU":  "vowel_au",
    # Alternate matra names (old newa_to_devanagari.py used _sign suffix)
    "aa_sign":   "matra_aa",
    "i_sign":    "matra_i",
    "ii_sign":   "matra_ii",
    "u_sign":    "matra_u",
    "uu_sign":   "matra_uu",
    "e_sign":    "matra_e",
    "ai_sign":   "matra_ai",
    "o_sign":    "matra_o",
    "au_sign":   "matra_au",
    # Other alternates
    "halant":       "virama",
    "chandrabindu": "candrabindu",
}


# ═══════════════════════════════════════════════════════════════════
# LOOKUP
# ═══════════════════════════════════════════════════════════════════

def get_char_info(class_name: str) -> dict:
    """Look up class name → Devanagari info. Case-insensitive."""
    name = class_name.strip()

    # 1. Direct lookup (exact case as stored)
    if name in CHAR_MAP:
        return CHAR_MAP[name]

    # 2. Lowercase direct lookup
    lower = name.lower()
    if lower in CHAR_MAP:
        return CHAR_MAP[lower]

    # 3. Alias lookup (handles vowel_A → vowel_a, o_sign → matra_o, etc.)
    if name in ALIASES:
        target = ALIASES[name]
        if target in CHAR_MAP:
            return CHAR_MAP[target]

    if lower in ALIASES:
        target = ALIASES[lower]
        if target in CHAR_MAP:
            return CHAR_MAP[target]

    return {"deva": "⟨?⟩", "iast": f"[{name}]", "desc": f"unmapped: {name}"}


def convert_sequence(class_names: list, output_format: str = "devanagari") -> str:
    result = []
    for name in class_names:
        info = get_char_info(name)
        result.append(info["iast"] if output_format == "iast" else info["deva"])
    return "".join(result)


def predictions_to_text(predictions: list, output_format: str = "devanagari",
                        low_conf_marker: str = "⟨?⟩") -> dict:
    lines_dict = {}
    flagged    = []

    for pred in predictions:
        line_idx   = pred.get("line", 0)
        class_name = pred.get("predicted", "space")
        is_low     = pred.get("low_conf", False)
        confidence = pred.get("confidence", 1.0)

        if is_low:
            char = low_conf_marker
            flagged.append({
                "file":       pred.get("file", ""),
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


def check_coverage(class_list: list) -> None:
    """Check all class names are mapped. Print ✓ or ✗ for each."""
    print(f"\nChecking coverage for {len(class_list)} classes...\n")
    unmapped = []
    for name in sorted(class_list):
        info   = get_char_info(name)
        ok     = "⟨?⟩" not in info["deva"]
        mark   = "✓" if ok else "✗ UNMAPPED"
        if not ok:
            unmapped.append(name)
        print(f"  {mark:12s} {name:25s} → {info['deva']}  ({info['iast']})")

    print(f"\n{'─'*55}")
    if unmapped:
        print(f"\n⚠  {len(unmapped)} still unmapped: {unmapped}")
    else:
        print(f"\n✓ All {len(class_list)} classes are mapped!")


# ═══════════════════════════════════════════════════════════════════
# SELF-TEST — run directly to verify all 67 classes
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json, pathlib

    # Load real class_map if available
    class_map_path = pathlib.Path("dataset_final/class_map.json")
    if class_map_path.exists():
        with open(class_map_path, encoding="utf-8") as f:
            cm = json.load(f)
        class_list = sorted(cm.keys())
        print(f"Loaded {len(class_list)} classes from {class_map_path}")
    else:
        # Fallback: all 67 known classes
        class_list = [
            "anusvara", "avagraha", "ba", "bha", "ca", "candrabindu",
            "cha", "da", "dda", "ddha", "dha",
            "digit_0", "digit_1", "digit_2", "digit_3", "digit_4",
            "digit_5", "digit_6", "digit_7", "digit_8", "digit_9",
            "ga", "gha", "ha", "ja", "jha", "ka", "kha", "la", "ma",
            "matra_aa", "matra_ai", "matra_au", "matra_e", "matra_i",
            "matra_ii", "matra_o", "matra_u", "matra_uu",
            "na", "nga", "nna", "nya", "pa", "pha", "ra", "sa", "sha",
            "ssa", "ta", "tha", "tta", "ttha", "virama", "visarga",
            "vowel_A", "vowel_AA", "vowel_AI", "vowel_AU", "vowel_E",
            "vowel_I", "vowel_II", "vowel_O", "vowel_U", "vowel_UU",
            "wa", "ya",
        ]

    check_coverage(class_list)

    print("\nSample Devanagari conversion:")
    seq = ["ka", "matra_aa", "la", "virama", "pa", "matra_aa", "danda"]
    print(f"  Input:       {seq}")
    print(f"  Devanagari:  {convert_sequence(seq, 'devanagari')}")
    print(f"  IAST:        {convert_sequence(seq, 'iast')}")

    print("\nVowel test:")
    seq2 = ["vowel_A", "vowel_AA", "vowel_I", "vowel_U", "vowel_E", "vowel_O"]
    print(f"  Input:       {seq2}")
    print(f"  Devanagari:  {convert_sequence(seq2, 'devanagari')}")

    print("\nMatra test:")
    seq3 = ["ka", "matra_i", "ta", "matra_aa", "ba", "matra_u", "dha"]
    print(f"  Input:       {seq3}")
    print(f"  Devanagari:  {convert_sequence(seq3, 'devanagari')}")