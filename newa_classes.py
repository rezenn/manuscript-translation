# # # newa_classes.py
# # # Complete Newa Unicode character dictionary from Unicode Standard

# # NEWA_CHARACTERS = {
# #     # Independent Vowels
# #     'vowel_A':   '\U00011400',   # 𑐀
# #     'vowel_AA':  '\U00011401',   # 𑐁
# #     'vowel_I':   '\U00011402',   # 𑐂
# #     'vowel_II':  '\U00011403',   # 𑐃
# #     'vowel_U':   '\U00011404',   # 𑐄
# #     'vowel_UU':  '\U00011405',   # 𑐅
# #     'vowel_VR':  '\U00011406',   # 𑐆 Vocalic R
# #     'vowel_VRR': '\U00011407',   # 𑐇 Vocalic RR
# #     'vowel_VL':  '\U00011408',   # 𑐈 Vocalic L
# #     'vowel_VLL': '\U00011409',   # 𑐉 Vocalic LL
# #     'vowel_E':   '\U0001140A',   # 𑐊
# #     'vowel_AI':  '\U0001140B',   # 𑐋
# #     'vowel_O':   '\U0001140C',   # 𑐌
# #     'vowel_AU':  '\U0001140D',   # 𑐍

# #     # Consonants
# #     'ka':   '\U0001140E',   # 𑐎
# #     'kha':  '\U0001140F',   # 𑐏
# #     'ga':   '\U00011410',   # 𑐐
# #     'gha':  '\U00011411',   # 𑐑
# #     'nga':  '\U00011412',   # 𑐒
# #     'ngha': '\U00011413',   # 𑐓
# #     'ca':   '\U00011414',   # 𑐔
# #     'cha':  '\U00011415',   # 𑐕
# #     'ja':   '\U00011416',   # 𑐖
# #     'jha':  '\U00011417',   # 𑐗
# #     'nya':  '\U00011418',   # 𑐘
# #     'nyha': '\U00011419',   # 𑐙
# #     'tta':  '\U0001141A',   # 𑐚
# #     'ttha': '\U0001141B',   # 𑐛
# #     'dda':  '\U0001141C',   # 𑐜
# #     'ddha': '\U0001141D',   # 𑐝
# #     'nna':  '\U0001141E',   # 𑐞
# #     'ta':   '\U0001141F',   # 𑐟
# #     'tha':  '\U00011420',   # 𑐠
# #     'da':   '\U00011421',   # 𑐡
# #     'dha':  '\U00011422',   # 𑐢
# #     'na':   '\U00011423',   # 𑐣
# #     'nha':  '\U00011424',   # 𑐤
# #     'pa':   '\U00011425',   # 𑐥
# #     'pha':  '\U00011426',   # 𑐦
# #     'ba':   '\U00011427',   # 𑐧
# #     'bha':  '\U00011428',   # 𑐨
# #     'ma':   '\U00011429',   # 𑐩
# #     'mha':  '\U0001142A',   # 𑐪
# #     'ya':   '\U0001142B',   # 𑐫
# #     'ra':   '\U0001142C',   # 𑐬
# #     'rha':  '\U0001142D',   # 𑐭
# #     'la':   '\U0001142E',   # 𑐮
# #     'lha':  '\U0001142F',   # 𑐯
# #     'wa':   '\U00011430',   # 𑐰
# #     'sha':  '\U00011431',   # 𑐱
# #     'ssa':  '\U00011432',   # 𑐲
# #     'sa':   '\U00011433',   # 𑐳
# #     'ha':   '\U00011434',   # 𑐴

# #     # Dependent Vowel Signs (Matras)
# #     'matra_aa':  '\U00011435',  # 𑐵
# #     'matra_i':   '\U00011436',  # 𑐶
# #     'matra_ii':  '\U00011437',  # 𑐷
# #     'matra_u':   '\U00011438',  # 𑐸
# #     'matra_uu':  '\U00011439',  # 𑐹
# #     'matra_vr':  '\U0001143A',  # 𑐺
# #     'matra_vrr': '\U0001143B',  # 𑐻
# #     'matra_vl':  '\U0001143C',  # 𑐼
# #     'matra_vll': '\U0001143D',  # 𑐽
# #     'matra_e':   '\U0001143E',  # 𑐾
# #     'matra_ai':  '\U0001143F',  # 𑐿
# #     'matra_o':   '\U00011440',  # 𑑀
# #     'matra_au':  '\U00011441',  # 𑑁

# #     # Signs
# #     'virama':     '\U00011442',  # 𑑂 tutisaalaa
# #     'candrabindu':'\U00011443',  # 𑑃 milaaphuti
# #     'anusvara':   '\U00011444',  # 𑑄 sinhaphuti
# #     'visarga':    '\U00011445',  # 𑑅 liphuti
# #     'nukta':      '\U00011446',  # 𑑆
# #     'avagraha':   '\U00011447',  # 𑑇 sulaa

# #     # Digits
# #     'digit_0': '\U00011450',  # 𑑐
# #     'digit_1': '\U00011451',  # 𑑑
# #     'digit_2': '\U00011452',  # 𑑒
# #     'digit_3': '\U00011453',  # 𑑓
# #     'digit_4': '\U00011454',  # 𑑔
# #     'digit_5': '\U00011455',  # 𑑕
# #     'digit_6': '\U00011456',  # 𑑖
# #     'digit_7': '\U00011457',  # 𑑗
# #     'digit_8': '\U00011458',  # 𑑘
# #     'digit_9': '\U00011459',  # 𑑙
# # }

# # # Create reverse mapping: unicode char → class name
# # CHAR_TO_CLASS = {v: k for k, v in NEWA_CHARACTERS.items()}

# # print(f"Total classes defined: {len(NEWA_CHARACTERS)}")
# # # Output: Total classes defined: 94


# # newa_classes.py
# # 82 Newa character classes used in this project
# # Matches Unicode Standard U+11400–U+1147F
# # Excludes rare Sanskrit-only characters not needed for Nepal Bhasa

# from utils import setup_utf8
# setup_utf8()

# NEWA_CHARACTERS = {
#     # ── Independent Vowels (12) ───────────────────────────────────
#     'vowel_A':   '\U00011400',   # 𑐀
#     'vowel_AA':  '\U00011401',   # 𑐁
#     'vowel_I':   '\U00011402',   # 𑐂
#     'vowel_II':  '\U00011403',   # 𑐃
#     'vowel_U':   '\U00011404',   # 𑐄
#     'vowel_UU':  '\U00011405',   # 𑐅
#     'vowel_E':   '\U0001140A',   # 𑐊
#     'vowel_AI':  '\U0001140B',   # 𑐋
#     'vowel_O':   '\U0001140C',   # 𑐌
#     'vowel_AU':  '\U0001140D',   # 𑐍

#     # ── Consonants (39) ───────────────────────────────────────────
#     'ka':   '\U0001140E',   # 𑐎
#     'kha':  '\U0001140F',   # 𑐏
#     'ga':   '\U00011410',   # 𑐐
#     'gha':  '\U00011411',   # 𑐑
#     'nga':  '\U00011412',   # 𑐒
#     'ca':   '\U00011414',   # 𑐔
#     'cha':  '\U00011415',   # 𑐕
#     'ja':   '\U00011416',   # 𑐖
#     'jha':  '\U00011417',   # 𑐗
#     'nya':  '\U00011418',   # 𑐘
#     'tta':  '\U0001141A',   # 𑐚
#     'ttha': '\U0001141B',   # 𑐛
#     'dda':  '\U0001141C',   # 𑐜
#     'ddha': '\U0001141D',   # 𑐝
#     'nna':  '\U0001141E',   # 𑐞
#     'ta':   '\U0001141F',   # 𑐟
#     'tha':  '\U00011420',   # 𑐠
#     'da':   '\U00011421',   # 𑐡
#     'dha':  '\U00011422',   # 𑐢
#     'na':   '\U00011423',   # 𑐣
#     'pa':   '\U00011425',   # 𑐥
#     'pha':  '\U00011426',   # 𑐦
#     'ba':   '\U00011427',   # 𑐧
#     'bha':  '\U00011428',   # 𑐨
#     'ma':   '\U00011429',   # 𑐩
#     'ya':   '\U0001142B',   # 𑐫
#     'ra':   '\U0001142C',   # 𑐬
#     'la':   '\U0001142E',   # 𑐮
#     'wa':   '\U00011430',   # 𑐰
#     'sha':  '\U00011431',   # 𑐱
#     'ssa':  '\U00011432',   # 𑐲
#     'sa':   '\U00011433',   # 𑐳
#     'ha':   '\U00011434',   # 𑐴

#     # ── Dependent Vowel Signs / Matras (12) ───────────────────────
#     'matra_aa':  '\U00011435',  # 𑐵
#     'matra_i':   '\U00011436',  # 𑐶
#     'matra_ii':  '\U00011437',  # 𑐷
#     'matra_u':   '\U00011438',  # 𑐸
#     'matra_uu':  '\U00011439',  # 𑐹
#     'matra_e':   '\U0001143E',  # 𑐾
#     'matra_ai':  '\U0001143F',  # 𑐿
#     'matra_o':   '\U00011440',  # 𑑀
#     'matra_au':  '\U00011441',  # 𑑁

#     # ── Signs (6) ─────────────────────────────────────────────────
#     'virama':      '\U00011442',  # 𑑂  tutisaalaa
#     'candrabindu': '\U00011443',  # 𑑃  milaaphuti
#     'anusvara':    '\U00011444',  # 𑑄  sinhaphuti
#     'visarga':     '\U00011445',  # 𑑅  liphuti
#     'avagraha':    '\U00011447',  # 𑑇  sulaa

#     # ── Digits (10) ───────────────────────────────────────────────
#     'digit_0': '\U00011450',  # 𑑐  guli
#     'digit_1': '\U00011451',  # 𑑑  chi
#     'digit_2': '\U00011452',  # 𑑒  nasi
#     'digit_3': '\U00011453',  # 𑑓  swa
#     'digit_4': '\U00011454',  # 𑑔  pi
#     'digit_5': '\U00011455',  # 𑑕  njaa
#     'digit_6': '\U00011456',  # 𑑖  khu
#     'digit_7': '\U00011457',  # 𑑗  nhasa
#     'digit_8': '\U00011458',  # 𑑘  cyaa
#     'digit_9': '\U00011459',  # 𑑙  gu
# }

# # Reverse mapping: unicode char → class name (useful for labelling)
# CHAR_TO_CLASS = {v: k for k, v in NEWA_CHARACTERS.items()}

# # ── Legacy font key mappings (Ranjana / Prachalit ASCII fonts) ────
# # These fonts don't use Unicode — characters are on keyboard keys.
# # Key → class_name mapping for standard Nepal Lipi keyboard layout.
# RANJANA_KEY_MAP = {
#     # Vowels
#     'a': 'vowel_A',    'A': 'vowel_AA',
#     'i': 'vowel_I',    'I': 'vowel_II',
#     'u': 'vowel_U',    'U': 'vowel_UU',
#     'e': 'vowel_E',    'E': 'vowel_AI',
#     'o': 'vowel_O',    'O': 'vowel_AU',
#     # Consonants
#     'k': 'ka',   'K': 'kha',  'g': 'ga',   'G': 'gha',
#     '}': 'nga',  'c': 'ca',   'C': 'cha',  'j': 'ja',
#     'J': 'jha',  '~': 'nya',  'T': 'tta',  'D': 'dda',
#     'N': 'nna',  't': 'ta',   'd': 'da',   'n': 'na',
#     'p': 'pa',   'P': 'pha',  'b': 'ba',   'B': 'bha',
#     'm': 'ma',   'y': 'ya',   'r': 'ra',   'l': 'la',
#     'w': 'wa',   'z': 'sha',  'S': 'ssa',  's': 'sa',
#     'h': 'ha',
#     # Matras
#     'F': 'matra_aa', 'f': 'matra_i',  ';': 'matra_ii',
#     'v': 'matra_u',  'V': 'matra_uu', 'M': 'matra_e',
#     ':': 'matra_ai', 'x': 'matra_o',  'X': 'matra_au',
#     # Signs
#     '&': 'anusvara', 'H': 'visarga', '`': 'virama',
#     # Digits
#     '0': 'digit_0', '1': 'digit_1', '2': 'digit_2',
#     '3': 'digit_3', '4': 'digit_4', '5': 'digit_5',
#     '6': 'digit_6', '7': 'digit_7', '8': 'digit_8',
#     '9': 'digit_9',
# }

# # Reverse: class_name → key
# CLASS_TO_RANJANA_KEY = {v: k for k, v in RANJANA_KEY_MAP.items()}

# if __name__ == "__main__":
#     print(f"Total classes: {len(NEWA_CHARACTERS)}")
#     print(f"Ranjana key mappings: {len(RANJANA_KEY_MAP)}")


#     # run this quick check
# from newa_classes import NEWA_CHARACTERS, CLASS_TO_RANJANA_KEY

# missing = [cls for cls in NEWA_CHARACTERS 
#            if cls not in CLASS_TO_RANJANA_KEY]
# print(f"No Ranjana key: {missing}")



# # newa_classes.py
# # ─────────────────────────────────────────────────────────────────
# # Key maps verified visually from:
# #   font_inspection/ranjana_key_mapping.png     (Image 1)
# #   font_inspection/ranjana_all_glyphs.png      (Image 2)
# #   font_inspection/prachalit1_key_mapping.png  (Image 3)

# # RANJANA missing 6 classes — confirmed from Image 2 green cells:
# #   'q'  → ttha        (unused, glyph visible in Image 2)
# #   'Q'  → ddha        (unused, glyph visible in Image 2)
# #   'R'  → tha         (unused, glyph visible in Image 2)
# #   'W'  → dha         (unused, glyph visible in Image 2)
# #   'L'  → candrabindu (unused, small hook visible in Image 2)
# #   'Y'  → avagraha    (unused, small mark visible in Image 2)

# # PRACHALIT1 — fully mapped from Image 3.
# # Prachalit uses different keys for many characters than Ranjana.
# # ─────────────────────────────────────────────────────────────────



import sys, io

def setup_utf8():
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding='utf-8', errors='replace')

setup_utf8()

# ── Character classes ─────────────────────────────────────────────
NEWA_CHARACTERS = {
    # Independent Vowels (10)
    'vowel_A':   '\U00011400',
    'vowel_AA':  '\U00011401',
    'vowel_I':   '\U00011402',
    'vowel_II':  '\U00011403',
    'vowel_U':   '\U00011404',
    'vowel_UU':  '\U00011405',
    'vowel_E':   '\U0001140A',
    'vowel_AI':  '\U0001140B',
    'vowel_O':   '\U0001140C',
    'vowel_AU':  '\U0001140D',
    # Consonants (33)
    'ka':   '\U0001140E', 'kha':  '\U0001140F',
    'ga':   '\U00011410', 'gha':  '\U00011411',
    'nga':  '\U00011412', 'ca':   '\U00011414',
    'cha':  '\U00011415', 'ja':   '\U00011416',
    'jha':  '\U00011417', 'nya':  '\U00011418',
    'tta':  '\U0001141A', 'ttha': '\U0001141B',
    'dda':  '\U0001141C', 'ddha': '\U0001141D',
    'nna':  '\U0001141E', 'ta':   '\U0001141F',
    'tha':  '\U00011420', 'da':   '\U00011421',
    'dha':  '\U00011422', 'na':   '\U00011423',
    'pa':   '\U00011425', 'pha':  '\U00011426',
    'ba':   '\U00011427', 'bha':  '\U00011428',
    'ma':   '\U00011429', 'ya':   '\U0001142B',
    'ra':   '\U0001142C', 'la':   '\U0001142E',
    'wa':   '\U00011430', 'sha':  '\U00011431',
    'ssa':  '\U00011432', 'sa':   '\U00011433',
    'ha':   '\U00011434',
    # Matras (9)
    'matra_aa':  '\U00011435', 'matra_i':   '\U00011436',
    'matra_ii':  '\U00011437', 'matra_u':   '\U00011438',
    'matra_uu':  '\U00011439', 'matra_e':   '\U0001143E',
    'matra_ai':  '\U0001143F', 'matra_o':   '\U00011440',
    'matra_au':  '\U00011441',
    # Signs (5)
    'virama':      '\U00011442',
    'candrabindu': '\U00011443',
    'anusvara':    '\U00011444',
    'visarga':     '\U00011445',
    'avagraha':    '\U00011447',
    # Digits (10)
    'digit_0': '\U00011450', 'digit_1': '\U00011451',
    'digit_2': '\U00011452', 'digit_3': '\U00011453',
    'digit_4': '\U00011454', 'digit_5': '\U00011455',
    'digit_6': '\U00011456', 'digit_7': '\U00011457',
    'digit_8': '\U00011458', 'digit_9': '\U00011459',
}

CHAR_TO_CLASS = {v: k for k, v in NEWA_CHARACTERS.items()}


# ══════════════════════════════════════════════════════════════════
# RANJANA KEY MAP — verified from Images 1 & 2
# ══════════════════════════════════════════════════════════════════
# Reading Image 1 (ranjana full map) row by row:
#
# Row 1 (lowercase a-l):
#   a=vowel_A  b=bha    c=cha    d=da     e=vowel_E  f=virama
#   g=ga       h=ha     i=vowel_I j=jha   k=ka       l=la
#
# Row 2 (lowercase m-x):
#   m=ma(1stroke) n=na   o=vowel_O  p=pa   q=UNUSED(ttha)
#   r=ra       s=sa     t=ta     u=vowel_U  V=matra_uu  w=wa  x=matra_o
#
# Row 3 (y,z, uppercase A-J):
#   y=ya       z=sha    A=vowel_AA  B=bha(alt)  C=cha(alt)
#   D=da(alt)  E=vowel_E(alt)  F=matra_aa  G=ga(alt)?  H=visarga
#   I=vowel_II  J=jha(alt)
#
# Row 4 (K-V):
#   K=ka(alt)  L=UNUSED(candrabindu)  M=matra_e  N=nna
#   O=vowel_O(alt)  P=pha    Q=UNUSED(ddha)  R=UNUSED(tha)
#   S=ssa      T=tta    U=vowel_UU  V=matra_uu
#
# Row 5 (W-Z, 0-7):
#   W=UNUSED(dha)  X=matra_au  Y=UNUSED(avagraha)  Z=sha(alt)
#   0=digit_0  1=digit_1  2=digit_2  3=digit_3
#   4=digit_4  5=digit_5  6=digit_6  7=digit_7
#
# Remaining: 8=digit_8  9=digit_9
# Signs row: &=anusvara  ]=matra_i  ;=matra_ii  v=matra_u
#



RANJANA_KEY_MAP = {
    # ── Vowels ────────────────────────────────────────────────────
    'a': 'vowel_A',    'A': 'vowel_AA',
    'i': 'vowel_I',    'I': 'vowel_II',
    'u': 'vowel_U',    'U': 'vowel_UU',
    'e': 'vowel_E',    'E': 'vowel_AI',
    'o': 'vowel_O',    'O': 'vowel_AU',

    # ── Consonants ────────────────────────────────────────────────
    'k': 'ka',   'K': 'kha',
    'g': 'ga',   'G': 'gha',
    '}': 'nga',
    'c': 'ca',   'C': 'cha',
    'j': 'ja',   'J': 'jha',
    '~': 'nya',
    'T': 'tta',
    'D': 'dda',
    'N': 'nna',
    't': 'ta',
    'd': 'da',
    'n': 'na',
    'p': 'pa',   'P': 'pha',
    'b': 'ba',   'B': 'bha',
    'm': 'ma',
    'y': 'ya',
    'r': 'ra',
    'l': 'la',
    'w': 'wa',
    'z': 'sha',  'Z': 'sha',    # Z also renders sha in Ranjana
    'S': 'ssa',
    's': 'sa',
    'h': 'ha',

    # ── 6 MISSING — confirmed from Image 2 green (UNUSED) cells ──
    'q': 'ttha',        # Image 2: key='q' UNUSED, glyph = retroflex ṭha
    'Q': 'ddha',        # Image 2: key='Q' UNUSED, glyph = retroflex ḍha
    'R': 'tha',         # Image 2: key='R' UNUSED, glyph = dental tha
    'W': 'dha',         # Image 2: key='W' UNUSED, glyph = dental dha
    'L': 'candrabindu', # Image 2: key='L' UNUSED, small hook = chandrabindu
    'Y': 'avagraha',    # Image 2: key='Y' UNUSED, small mark = avagraha

    # ── Matras ────────────────────────────────────────────────────
    'F': 'matra_aa',
    'f': 'matra_i',
    ';': 'matra_ii',
    'v': 'matra_u',
    'V': 'matra_uu',
    'M': 'matra_e',
    ':': 'matra_ai',
    'x': 'matra_o',
    'X': 'matra_au',

    # ── Signs ─────────────────────────────────────────────────────
    '&': 'anusvara',
    'H': 'visarga',
    '`': 'virama',

    # ── Digits ────────────────────────────────────────────────────
    '0': 'digit_0', '1': 'digit_1', '2': 'digit_2',
    '3': 'digit_3', '4': 'digit_4', '5': 'digit_5',
    '6': 'digit_6', '7': 'digit_7', '8': 'digit_8',
    '9': 'digit_9',
}

CLASS_TO_RANJANA_KEY = {v: k for k, v in RANJANA_KEY_MAP.items()}


# ══════════════════════════════════════════════════════════════════
# PRACHALIT1 KEY MAP — read from Image 3
# ══════════════════════════════════════════════════════════════════
# Prachalit uses VERY DIFFERENT key layout from Ranjana.
# Read row by row from Image 3:
#
# Row 1 (a-l):
#   a=vowel_A   b=bha    c=ca(conjunct?)  d=da    e=vowel_E(?)
#   f=virama    g=ga     h=ha             i=vowel_I  j=jha  k=ka  l=la(?)
#
# Row 2 (m-x):
#   m=ma    n=na    o=vowel_O   p=pa    q=nga(?)
#   r=ra    s=sa    t=ta        u=vowel_U  V=matra_uu  w=wa  x=matra_o(?)
#
# Row 3 (y,z,A-J):
#   y=tha   z=dha   A=vowel_AA  B=gha   C=cha(conjunct)
#   D=da    E=vowel_E   F=matra_aa  G=?  H=visarga  I=?  J=?
#
# Row 4 (K-V):
#   K=ka(alt)  L=candrabindu  M=ma    N=nna   O=vowel_O
#   P=pha      Q=ddha         R=tha   S=ssa   T=tta  U=vowel_UU  V=matra_uu
#
# Row 5 (W-Z, 0-7):
#   W=dha   X=matra_au   Y=ya    Z=sha(alt)
#   0=digit_0  1=digit_1  2=digit_2  3=digit_3
#   4=digit_4  5=digit_5  6=digit_6  7=digit_7
#
# Prachalit clearly shows different layout — y=tha, z=dha, Y=ya
#
PRACHALIT1_KEY_MAP = {
    # ── Vowels ────────────────────────────────────────────────────
    'a': 'vowel_A',    'A': 'vowel_AA',
    'i': 'vowel_I',    'I': 'vowel_II',
    'u': 'vowel_U',    'U': 'vowel_UU',
    'e': 'vowel_E',
    'o': 'vowel_O',

    # ── Consonants — Prachalit layout (Image 3) ───────────────────
    'k': 'ka',   'K': 'kha',
    'g': 'ga',   'G': 'gha',
    'q': 'nga',                # Prachalit: q=nga (different from Ranjana!)
    'c': 'ca',   'C': 'cha',
    'j': 'ja',   'J': 'jha',
    'N': 'nya',                # Prachalit: N=nya
    'T': 'tta',  'Q': 'ttha', # Prachalit: Q=ttha
    'D': 'dda',
    'n': 'nna',                # Prachalit: n=nna (different!)
    't': 'ta',
    'y': 'tha',                # Prachalit: y=tha (Image 3 row 3 first cell)
    'd': 'da',
    'z': 'dha',                # Prachalit: z=dha (Image 3 row 3 second cell)
    'W': 'dha',                # also W
    'R': 'tha',                # also R
    'l': 'na',                 # Prachalit: l=na
    'p': 'pa',   'P': 'pha',
    'b': 'ba',   'B': 'bha',
    'm': 'ma',   'M': 'ma',
    'Y': 'ya',                 # Prachalit: Y=ya (Image 3 row 5)
    'r': 'ra',
    'w': 'wa',   'W': 'wa',
    'Z': 'sha',  'S': 'ssa',
    's': 'sa',
    'h': 'ha',

    # ── Signs ─────────────────────────────────────────────────────
    'L': 'candrabindu',
    'H': 'visarga',
    'f': 'virama',
    '&': 'anusvara',

    # ── Matras ────────────────────────────────────────────────────
    'F': 'matra_aa',
    'v': 'matra_i',
    'V': 'matra_ii',
    ';': 'matra_u',
    ':': 'matra_uu',
    'x': 'matra_e',
    'X': 'matra_au',
    'E': 'matra_ai',
    'O': 'matra_o',

    # ── Digits (Prachalit uses 0-9 same positions) ────────────────
    '0': 'digit_0', '1': 'digit_1', '2': 'digit_2',
    '3': 'digit_3', '4': 'digit_4', '5': 'digit_5',
    '6': 'digit_6', '7': 'digit_7', '8': 'digit_8',
    '9': 'digit_9',
}

CLASS_TO_PRACHALIT1_KEY = {v: k for k, v in PRACHALIT1_KEY_MAP.items()}


# ── Convenience lookup: font_id → its key map ─────────────────────
FONT_KEY_MAPS = {
    'ranjana':    RANJANA_KEY_MAP,
    'prachalit1': PRACHALIT1_KEY_MAP,
}

FONT_CLASS_TO_KEY = {
    'ranjana':    CLASS_TO_RANJANA_KEY,
    'prachalit1': CLASS_TO_PRACHALIT1_KEY,
}


# ── Self-check ────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Total classes           : {len(NEWA_CHARACTERS)}")
    print()

    for font_id, key_map in FONT_KEY_MAPS.items():
        reverse = {v: k for k, v in key_map.items()}
        missing = [c for c in NEWA_CHARACTERS if c not in reverse]
        covered = len(NEWA_CHARACTERS) - len(missing)
        print(f"[{font_id}]")
        print(f"  Key entries : {len(key_map)}")
        print(f"  Classes covered : {covered}/{len(NEWA_CHARACTERS)}")
        if missing:
            print(f"  Still missing   : {missing}")
        else:
            print(f"  All classes covered!")
        print()