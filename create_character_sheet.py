# create_character_sheet.py
from PIL import Image, ImageDraw, ImageFont
from newa_classes import NEWA_CHARACTERS

# FONT_PATH = "fonts/NotoSansNewa-Regular.ttf"
FONT_PATH = "fonts/NithyaRanjanaNU-Regular.otf"

# Layout settings
GRID_COLS = 5          # 5 characters per row (more space)
REF_SIZE = 80          # size of reference character box
WRITE_BOXES = 5        # how many boxes writer fills per character
BOX_SIZE = 70          # size of each writing box
ROW_HEIGHT = 130       # height per row
COL_WIDTH = REF_SIZE + (WRITE_BOXES * (BOX_SIZE + 5)) + 30
PADDING = 20

chars = list(NEWA_CHARACTERS.items())
rows = (len(chars) + GRID_COLS - 1) // GRID_COLS

sheet_w = GRID_COLS * COL_WIDTH + PADDING * 2
sheet_h = rows * ROW_HEIGHT + PADDING * 2 + 60  # 60 for title

sheet = Image.new('RGB', (sheet_w, sheet_h), 'white')
draw = ImageDraw.Draw(sheet)

# Load fonts
try:
    newa_font  = ImageFont.truetype(FONT_PATH, 48)
except:
    newa_font  = ImageFont.load_default()

try:
    label_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 11)
    title_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
except:
    label_font = ImageFont.load_default()
    title_font = label_font

# Title
draw.text(
    (PADDING, PADDING),
    "Thesis Data Collection | Newa Script Handwriting Collection Sheet  |  "
    "Writer Name: _________________________  Date: ______________",
    fill='black', font=title_font
)

# Draw each character row
for i, (class_name, char) in enumerate(chars):
    col = i % GRID_COLS
    row = i // GRID_COLS

    base_x = PADDING + col * COL_WIDTH
    base_y = PADDING + 60 + row * ROW_HEIGHT

    # --- Reference box (shaded) ---
    draw.rectangle(
        [base_x, base_y, base_x + REF_SIZE, base_y + REF_SIZE],
        fill='#f0f0f0', outline='black', width=2
    )

    # Draw the Newa character centered in reference box
    try:
        bbox = draw.textbbox((0, 0), char, font=newa_font)
        cw = bbox[2] - bbox[0]
        ch = bbox[3] - bbox[1]
        cx = base_x + (REF_SIZE - cw) // 2 - bbox[0]
        cy = base_y + (REF_SIZE - ch) // 2 - bbox[1]
        draw.text((cx, cy), char, fill='#000080', font=newa_font)
    except:
        draw.text((base_x + 5, base_y + 5), "?",
                  fill='red', font=label_font)

    # Class name label under reference box
    draw.text(
        (base_x, base_y + REF_SIZE + 3),
        class_name[:10],
        fill='#555555', font=label_font
    )

    # --- Empty writing boxes ---
    for b in range(WRITE_BOXES):
        bx = base_x + REF_SIZE + 10 + b * (BOX_SIZE + 5)
        by = base_y + (REF_SIZE - BOX_SIZE) // 2  # vertically center

        draw.rectangle(
            [bx, by, bx + BOX_SIZE, by + BOX_SIZE],
            fill='white', outline='#aaaaaa', width=1
        )
        # Box number (faint)
        draw.text(
            (bx + 2, by + 2),
            str(b + 1),
            fill='#cccccc', font=label_font
        )

# Save
sheet.save("character_sheet_ranjana.pdf", "PDF",  resolution=150)
print(f"Saved character_sheet_ranjana.pdf")
print(f"Sheet size: {sheet_w} x {sheet_h} px")
print(f"Total characters: {len(chars)}")
print(f"Writing boxes per character: {WRITE_BOXES}")
print(f"Total handwriting samples when filled: "
      f"{len(chars) * WRITE_BOXES} per writer")