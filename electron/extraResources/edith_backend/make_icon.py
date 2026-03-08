#!/usr/bin/env python3
from pathlib import Path
import subprocess
import tempfile
import shutil

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
ICNS = ASSETS / "edith.icns"
ICON_PNG = ASSETS / "edith_icon.png"

ASSETS.mkdir(parents=True, exist_ok=True)

base = Image.new("RGBA", (1024, 1024), (244, 242, 236, 255))
draw = ImageDraw.Draw(base)

draw.rounded_rectangle(
    (120, 120, 904, 904),
    radius=180,
    fill=(255, 255, 255, 255),
    outline=(216, 212, 203, 255),
    width=12,
)

font = None
for candidate in [
    "/System/Library/Fonts/SFNSDisplay-Bold.otf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]:
    try:
        font = ImageFont.truetype(candidate, 460)
        break
    except Exception:
        continue
if font is None:
    font = ImageFont.load_default()

text = "E"
bbox = draw.textbbox((0, 0), text, font=font)
text_w = bbox[2] - bbox[0]
text_h = bbox[3] - bbox[1]
x = (1024 - text_w) // 2
y = (1024 - text_h) // 2 - 20

draw.text((x, y), text, fill=(28, 28, 28, 255), font=font)

icon_specs = [
    ("icon_16x16.png", 16),
    ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32),
    ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128),
    ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256),
    ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512),
    ("icon_512x512@2x.png", 1024),
]

with tempfile.TemporaryDirectory(prefix="edith_icon_") as tmp:
    tmp_path = Path(tmp)
    iconset = tmp_path / "edith.iconset"
    iconset.mkdir(parents=True, exist_ok=True)

    for filename, size in icon_specs:
        out = base.resize((size, size), Image.Resampling.LANCZOS)
        out.save(iconset / filename)

    base.save(ICON_PNG)
    tmp_icns = tmp_path / "edith.icns"
    subprocess.run(["xattr", "-cr", str(iconset)], check=False)
    try:
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(tmp_icns)], check=True)
        shutil.copy2(tmp_icns, ICNS)
        print(f"Created {ICNS}")
    except Exception:
        print("Warning: iconutil failed to create .icns, using PNG icon fallback.")

print(f"Created {ICON_PNG}")
