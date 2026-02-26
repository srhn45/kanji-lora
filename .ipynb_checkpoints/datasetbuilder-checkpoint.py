from lxml import etree
from collections import defaultdict
from pathlib import Path
import json
import re

def load_kanjidic(path):
    tree = etree.parse(path)
    root = tree.getroot()

    kanji_map = {}

    for char in root.findall("character"):
        literal = char.findtext("literal")

        meanings = []
        rm = char.find("reading_meaning")
        if rm is not None:
            for meaning in rm.findall(".//meaning"):
                if meaning.get("m_lang") in (None, "en"):
                    meanings.append(meaning.text)

        if meanings:
            kanji_map[literal] = meanings

    return kanji_map

def filename_to_kanji(stem):
    m = re.match(r"^([0-9a-fA-F]+)", stem)
    if not m:
        raise ValueError(f"Invalid kanjivg filename: {stem}")

    hexcode = m.group(1)
    return chr(int(hexcode, 16))


def build_dataset(png_dir, kanjidic):
    samples = []

    for img_path in Path(png_dir).glob("*.png"):
        kanji = filename_to_kanji(img_path.stem)

        if kanji not in kanjidic:
            continue

        for meaning in kanjidic[kanji]:
            meaning = meaning.strip()
            if not meaning:
                continue

            samples.append({
                "image": str(img_path),
                "text": meaning
            })

    return samples


def write_jsonl(samples, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

kanjidic = load_kanjidic("kanjidic2.xml")
samples = build_dataset("kanjivg_png", kanjidic)
write_jsonl(samples, "metadata.jsonl")

print(f"Samples: {len(samples)}")

for s in samples[:5]:
    print(s["image"], "→", s["text"])