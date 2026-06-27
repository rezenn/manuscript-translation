"""
postprocess.py  --  Sequence-level post-processing for NewaConvNet output
==========================================================================

PURPOSE
-------
Two problems arise on real manuscript images:

  1. ATTRACTOR BIAS -- certain classes attract uncertain predictions.
     From confusion matrix + per-class metrics the main attractors are:
       kha  (precision 86.4%, recall 98.6% -- fires on uncertain blobs)
       na   (precision 88.3%, recall 98.0% -- fires on small ink)
       nna  (NN -- rare in manuscripts but fires often)
       ga   (ga -- shows up more than expected)
       virama (precision 88.0% -- fires on small ink fragments)

     FIX: each attractor class has an ABSOLUTE minimum confidence
     threshold.  If the model fires on it below that threshold, it is
     suppressed (low_conf=True) regardless of the user's slider.

  2. GRAMMATICALLY IMPOSSIBLE SEQUENCES -- e.g. matra after matra,
     virama after virama, consonant cluster rule violations, 3+
     consecutive identical characters.

     FIX: sequence_repair() removes these physically impossible patterns.
"""

from __future__ import annotations
from typing import List, Dict, Any

# Absolute minimum confidence per attractor class.
# Below this threshold the character is suppressed even if above
# the user's global slider -- because these classes fire too often.
# Derived from: higher bar = rarer/more over-predicted the class is.
ATTRACTOR_MIN: Dict[str, float] = {
    "nna":     0.72,   # (NN) very rare in manuscripts; almost always wrong
    "kha":     0.65,   # precision 86.4% -- biggest attractor
    "virama":  0.62,   # precision 88.0% -- fires on small ink fragments
    "na":      0.60,   # precision 88.3% -- over-fires on many shapes
    "vowel_U": 0.60,   # precision 88.3%
    "wa":      0.60,   # precision 88.1%
    "ga":      0.58,   # shows up more than expected in manuscript
    "da":      0.55,   # precision 90.1%
    "ka":      0.55,   # precision 90.7%
    "ba":      0.55,   # precision 90.7%
    "pa":      0.52,   # precision 92.4%
}

MATRAS = frozenset({
    "matra_aa", "matra_i", "matra_ii", "matra_u", "matra_uu",
    "matra_e", "matra_ai", "matra_o", "matra_au",
})
VIRAMA = "virama"
SPACE  = "space"

CONSONANTS = frozenset({
    "ka","kha","ga","gha","nga","ca","cha","ja","jha","nya",
    "tta","ttha","dda","ddha","nna","ta","tha","da","dha","na",
    "pa","pha","ba","bha","ma","ya","ra","la","wa","sha","ssa","sa","ha",
})


def confidence_gate(
    char_list: List[Dict[str, Any]],
    global_threshold: float,
) -> List[Dict[str, Any]]:
    """
    Apply per-class absolute minimum thresholds for attractor classes.
    Characters failing their class threshold are flagged low_conf=True.
    Does NOT remove characters -- only adjusts the flag.
    """
    updated = []
    for c in char_list:
        c = dict(c)
        pred = c.get("predicted", "")
        conf = c.get("confidence", 0.0)

        # Skip space entries
        if pred == SPACE or c.get("file") == "__space__":
            updated.append(c)
            continue

        # Apply per-class absolute floor
        if pred in ATTRACTOR_MIN:
            floor = ATTRACTOR_MIN[pred]
            effective = max(global_threshold, floor)
            if conf < effective:
                c["low_conf"] = True
                c["postprocess_note"] = (
                    f"attractor_gate:{pred} conf={conf:.2f} "
                    f"< floor={effective:.2f}"
                )

        updated.append(c)
    return updated


def sequence_repair(
    char_list: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Remove physically impossible character sequences per Prachalit grammar.

    Rules applied per line (spaces are treated as syllable boundaries):
      R1  Matra/virama at line start (no consonant host) -> drop
      R2  Matra after matra -> drop second
      R3  Virama after virama -> drop second
      R4  Virama after matra -> drop virama
      R5  Matra after space/start -> drop (no host consonant)
      R6  3+ consecutive identical non-space chars -> keep max 2
    """
    # Group by line, preserving order
    lines: Dict[int, List[Dict]] = {}
    for c in char_list:
        ln = c.get("line", 0)
        lines.setdefault(ln, []).append(c)

    result: List[Dict] = []

    for ln in sorted(lines):
        chars = lines[ln]
        cleaned: List[Dict] = []

        # Track last TWO non-space predictions for R6
        recent: List[str] = []  # ring buffer of last 2 non-space preds

        for c in chars:
            pred      = c.get("predicted", "")
            is_space  = (pred == SPACE or c.get("file") == "__space__")
            is_matra  = pred in MATRAS
            is_virama = pred == VIRAMA

            prev_c    = cleaned[-1] if cleaned else None
            prev_pred = prev_c.get("predicted", "") if prev_c else ""
            prev_space  = (prev_pred == SPACE or
                           (prev_c.get("file") == "__space__" if prev_c else False))
            prev_matra  = prev_pred in MATRAS
            prev_virama = prev_pred == VIRAMA

            # R1: diacritic at line start
            if not cleaned and (is_matra or is_virama):
                c = dict(c); c["low_conf"] = True
                c["postprocess_note"] = "R1:diacritic_at_start->dropped"
                continue

            # R2: matra after matra
            if is_matra and prev_matra:
                c = dict(c); c["low_conf"] = True
                c["postprocess_note"] = "R2:matra_after_matra->dropped"
                continue

            # R3: virama after virama
            if is_virama and prev_virama:
                c = dict(c); c["low_conf"] = True
                c["postprocess_note"] = "R3:virama_after_virama->dropped"
                continue

            # R4: virama after matra
            if is_virama and prev_matra:
                c = dict(c); c["low_conf"] = True
                c["postprocess_note"] = "R4:virama_after_matra->dropped"
                continue

            # R5: matra/virama after space (no host consonant in this word)
            if (is_matra or is_virama) and prev_space:
                c = dict(c); c["low_conf"] = True
                c["postprocess_note"] = "R5:diacritic_after_space->dropped"
                continue

            # R6: 3+ consecutive identical non-space chars
            if not is_space and len(recent) >= 2 and recent[-1] == recent[-2] == pred:
                c = dict(c); c["low_conf"] = True
                c["postprocess_note"] = f"R6:triple_repeat_{pred}->dropped"
                continue

            # Accept this character
            cleaned.append(c)
            if not is_space:
                recent.append(pred)
                if len(recent) > 2:
                    recent.pop(0)

        result.extend(cleaned)

    return result


def postprocess(
    char_list: List[Dict[str, Any]],
    global_threshold: float = 0.55,
) -> List[Dict[str, Any]]:
    """
    Full post-processing pipeline:
      1. confidence_gate  -- per-class attractor floor thresholds
      2. sequence_repair  -- grammar-level impossible sequence removal
    """
    char_list = confidence_gate(char_list, global_threshold)
    char_list = sequence_repair(char_list)
    return char_list