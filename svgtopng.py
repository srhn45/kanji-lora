import cairosvg
import numpy as np
from PIL import Image
from lxml import etree
from pathlib import Path

def strip_stroke_numbers(svg_path, out_path):
    parser = etree.XMLParser(remove_comments=True)
    tree = etree.parse(str(svg_path), parser)
    root = tree.getroot()

    # SVG namespace handling
    ns = {"svg": root.nsmap.get(None)}

    # Remove all <text> elements (stroke numbers)
    for text in root.xpath(".//svg:text", namespaces=ns):
        text.getparent().remove(text)

    tree.write(str(out_path))


def svg_to_binary_png(svg_path, png_path, size=256):
    cairosvg.svg2png(
        url=str(svg_path),
        write_to=str(png_path),
        output_width=size,
        output_height=size,
        background_color="white"
    )

    img = Image.open(png_path).convert("L")
    arr = np.array(img)

    # Hard threshold
    binary = (arr > 128).astype(np.uint8) * 255
    Image.fromarray(binary, mode="L").save(png_path)

raw_svg_dir = Path("./kanji/kanji")
clean_svg_dir = Path("kanjivg_clean_svg")
png_dir = Path("kanjivg_png")

clean_svg_dir.mkdir(exist_ok=True)
png_dir.mkdir(exist_ok=True)

for svg in raw_svg_dir.glob("*.svg"):
    clean_svg = clean_svg_dir / svg.name
    png = png_dir / (svg.stem + ".png")

    strip_stroke_numbers(svg, clean_svg)
    svg_to_binary_png(clean_svg, png)
