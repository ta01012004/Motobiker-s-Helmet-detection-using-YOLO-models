# rename_voc_classes.py
import os, glob
import xml.etree.ElementTree as ET

ANN_DIR = r"E:\Deep\helmetdata\annotations"

MAPPING = {
    "With Helmet": "helmet",
    "with helmet": "helmet",
    "Helmet": "helmet",
    "helmet": "helmet",
    "Without Helmet": "no-helmet",
    "without helmet": "no-helmet",
    "No Helmet": "no-helmet",
    "no helmet": "no-helmet",
}

for xp in glob.glob(os.path.join(ANN_DIR, "*.xml")):
    tree = ET.parse(xp)
    root = tree.getroot()
    changed = False
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()
        if name in MAPPING and MAPPING[name] != name:
            obj.find("name").text = MAPPING[name]
            changed = True
    if changed:
        tree.write(xp, encoding="utf-8")
print("✅ Done: chuẩn hoá tên lớp trong XML.")
