"""Turn the long result files of eval_real.py into one workbook for humans.

Reads results/real_26-11-04/objects.csv, frames.csv and pitch_sensitivity.txt (run eval_real.py,
pitch_sensitivity.py and depth_to_excel.py first) and writes results/real_26-11-04/Messungen.xlsx:

  Messungen    one row per measured object: camera image with the object outlined, tape distance,
               estimate and error of every method, and a link to the depth workbook of that recording
  Überblick    mean distance error in % per object group and method (truth = tape measure)
  Boden+Wand   mean relative error of the floor (reference = ground plane from TF) and the wall (tape)
  Kopfneigung  how far the ball distance moves when the head pitch from TF is off by 1 degree

DA3 is scaled with the floor in front of the robot. Every bag counts once: first the median over its
frames, then the mean over the bags.

Usage: python scripts/make_tables.py
"""

import csv
import io
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import cv2
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font

sys.path.insert(0, str(Path(__file__).parent))
from depth_to_excel import EXTRA_FRAMES, excel_name, marked_image  # noqa: E402
from eval_real import load_frames  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/real_26-11-04"
# DA3 Small and Base are relative: they only give metres after scaling with the floor. Metric-Large is shown both
# scaled and raw (as the model outputs it).
ROW_METHODS = {"IPM": "IPM (Bit-Bots)", "ZED": "ZED", "Small skaliert": "DA3 Small skaliert",
               "Base skaliert": "DA3 Base skaliert", "Metric-Large skaliert": "DA3 Metric-Large skaliert",
               "Metric-Large roh": "DA3 Metric-Large roh"}
THUMB_W = 300  # px width of the image in the Messungen sheet
METHODS = ROW_METHODS
NOTE = {"": "", "mostly hidden": " (fast verdeckt)", "half hidden": " (halb verdeckt)", "in the air": " (in der Luft)"}


def display(label, note):
    """Label without image number and with a German note, e.g. 'Ball 1.5 m (in der Luft)'."""
    base = re.sub(r" \(.*?\)| Bild \d+", "", label)
    return base + NOTE[note]


def per_bag_objects():
    """(kind, label as in objects.csv, note, method) -> list of (truth, per-bag median estimate).
    Same weighting as summary.txt: one value per object and bag."""
    frames = defaultdict(list)
    for r in csv.DictReader(open(RES / "objects.csv")):
        frames[r["kind"], r["label"], r["note"], r["bag"], r["method"]].append((float(r["truth"]), float(r["estimate"])))
    out = defaultdict(list)
    for (kind, label, note, _bag, method), vals in frames.items():
        out[kind, label, note, method].append((vals[0][0], median(v[1] for v in vals)))
    return out


def style(ws, widths, percent_cols=(), cm_cols=()):
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[chr(64 + i)].width = w
    ws.freeze_panes = "B2"
    last = ws.max_row
    for col in percent_cols:
        rng = f"{col}2:{col}{last}"
        for row in ws[rng]:
            row[0].number_format = "0.0%"
    for col in cm_cols:
        for row in ws[f"{col}2:{col}{last}"]:
            row[0].number_format = "+0;-0;0"


def color(ws, cols, rows, lo_good=True):
    for col in cols:
        ws.conditional_formatting.add(f"{col}{rows[0]}:{col}{rows[1]}", ColorScaleRule(
            start_type="num", start_value=0, start_color="C6EFCE",
            end_type="percentile", end_value=90, end_color="FFC7CE"))


def thumbnail(d, only):
    """Camera image with object `only` outlined, scaled to THUMB_W pixels width, for embedding in a cell."""
    png = marked_image(d, only).getvalue()
    img = cv2.imdecode(__import__("numpy").frombuffer(png, "uint8"), cv2.IMREAD_COLOR)
    h = round(img.shape[0] * THUMB_W / img.shape[1])
    ok, out = cv2.imencode(".png", cv2.resize(img, (THUMB_W, h), interpolation=cv2.INTER_AREA))
    return XLImage(io.BytesIO(out.tobytes())), h


def measurements_sheet(ws):
    """One row per (bag, object): image, truth, estimate and error per method, link to the depth workbook."""
    ws.title = "Messungen"
    per = defaultdict(list)  # (bag, object) -> rows of objects.csv
    for r in csv.DictReader(open(RES / "objects.csv")):
        per[r["bag"], int(r["object"])].append(r)
    frames = {(d["bag"], d["file"]): d for d in load_frames()}
    header = ["Bild (Objekt umrandet)", "Messung", "Nr.", "Objekt", "Maßband ab Fußspitze (m)",
              "Wahr ab Robotermitte (m)"]
    header += [n for name in ROW_METHODS.values() for n in (name, "Fehler")] + ["Bilder", "Tiefenkarten"]
    ws.append(header)
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    row, last_bag = 2, None
    for (bag, i_obj), rows in sorted(per.items(), key=lambda kv: (float(kv[1][0]["truth"]), kv[0])):
        truth = float(rows[0]["truth"])
        short = bag.removeprefix("mde_s1_")
        kind = "Ball" if rows[0]["kind"] == "ball" else "Roboter"
        cells = [None, short, i_obj + 1, kind + NOTE[rows[0]["note"]], float(rows[0]["tape"]), round(truth, 2)]
        n_frames = len({r["file"] for r in rows})
        for m in ROW_METHODS:
            ests = [float(r["estimate"]) for r in rows if r["method"] == m and r["estimate"] not in ("", "nan")]
            e = median(ests) if ests else None
            cells += [round(e, 2) if e is not None else None, (e - truth) / truth if e is not None else None]
        cells += [n_frames, "öffnen"]
        ws.append(cells)
        for c, name in enumerate(header, start=1):
            if name == "Fehler":
                ws.cell(row, c).number_format = "+0.0%;-0.0%;0.0%"
            elif 5 <= c and name != "Bilder" and c < len(header):
                ws.cell(row, c).number_format = "0.00"  # distances in metres
            ws.cell(row, c).alignment = Alignment(vertical="center", wrap_text=True)
        link = ws.cell(row, len(header))
        idx = int(min(r["file"] for r in rows).split("_")[1].split(".")[0])
        name = excel_name(bag, idx if idx in EXTRA_FRAMES.get(bag, ()) else None)  # the file with this frame, if any
        link.hyperlink = f"../tiefe_excel_26-11-04/{name}"
        link.font = Font(color="0563C1", underline="single")
        # The first frame in which this object is marked; in some bags (mensch) each position has its own frame
        img, h = thumbnail(frames[bag, min(r["file"] for r in rows)], i_obj)
        ws.add_image(img, f"A{row}")
        ws.row_dimensions[row].height = h * 0.75 + 4  # points
        row += 1
    err_cols = [chr(64 + c) for c, name in enumerate(header, start=1) if name == "Fehler"]
    for col in err_cols:
        ws.conditional_formatting.add(f"{col}2:{col}{row - 1}", ColorScaleRule(
            start_type="num", start_value=-0.15, start_color="F8696B", mid_type="num", mid_value=0,
            mid_color="C6EFCE", end_type="num", end_value=0.15, end_color="F8696B"))
    ws.column_dimensions["A"].width = THUMB_W / 7
    ws.column_dimensions["B"].width = 18
    ws.column_dimensions["D"].width = 22
    ws.freeze_panes = "B2"
    ws.cell(row + 1, 2, "Wahr ab Robotermitte = Maßband + 9,9 cm: gemessen wurde ab der Fußspitze, verglichen wird ab base_footprint (Punkt zwischen den Füßen, 9,9 cm hinter der Fußspitze), wie im IPM der Bit-Bots. Werte = Median über die Bilder der "
                        "Messung. Fehler: + = zu weit, − = zu nah, grün = gut, rot = 15 % oder mehr daneben. "
                        "skaliert = DA3 über den Boden 0,5–2,5 m vor dem Roboter auf Meter gebracht, roh = Metric-Large "
                        "direkt aus dem Modell (Small und Base gibt es nur skaliert, sie liefern ohne Skalierung keine "
                        "Meter). „öffnen“ = Tiefe pro Pixel "
                        "für ZED und alle DA3-Modelle (Ordner tiefe_excel_26-11-04).")


def main():
    objs = per_bag_objects()
    wb = Workbook()
    measurements_sheet(wb.active)

    # Overview: mean |error| / truth per group
    ws = wb.create_sheet("Überblick")
    ws.append(["Mittlerer Fehler beim Abstand (Wahrheit = Maßband)"] + list(METHODS.values()) + ["Werte"])
    groups = [("Ball am Boden bis 1,5 m", "ball", lambda n, t: n == "" and t < 1.5),
              ("Ball am Boden 1,5–3,5 m", "ball", lambda n, t: n == "" and 1.5 <= t < 3.5),
              ("Ball am Boden 4,5 m", "ball", lambda n, t: n == "" and t >= 3.5),
              ("Roboter (frei)", "robot", lambda n, t: n == ""),
              ("Ball fast verdeckt", "ball", lambda n, t: n == "mostly hidden"),
              ("Roboter halb verdeckt", "robot", lambda n, t: n == "half hidden"),
              ("Ball in der Luft", "ball", lambda n, t: n == "in the air")]
    for name, kind, keep in groups:
        row, n = [name], 0
        for m in METHODS:
            errs = [abs(e - t) / t for (k, _l, note, mm), vals in objs.items() if k == kind and mm == m
                    for t, e in vals if keep(note, t)]
            row.append(mean(errs) if errs else None)
            n = len(errs)
        ws.append(row + [n])
    style(ws, [34, 14, 12, 14, 14, 16, 16, 8], percent_cols="BCDEFG")
    color(ws, "BCDEFG", (2, 5))
    ws.append([])
    ws.append(["Lesart: 3 % heißt, bei 2 m liegt die Schätzung im Mittel 6 cm daneben. skaliert = pro Bild über den "
               "Boden 0,5–2,5 m vor dem Roboter auf Meter gebracht, roh = Metric-Large direkt aus dem Modell. Abstände waagerecht ab base_footprint, Maßband + 9,9 cm."])
    ws.append(["Daten: Aufnahme 04.11., ein Raum, 21 Bags. Ballradius noch nicht gemessen (7,65 cm wie im IPM)."])

    # Floor and wall: flat head, scenes without robots or people (same selection as summary.txt)
    rows = list(csv.DictReader(open(RES / "frames.csv")))
    cluttered = ("offen", "verdeckt", "ball_vorne", "ball_hinten", "mensch")
    bag_vals = defaultdict(list)
    for r in rows:
        if any(c in r["bag"] for c in cluttered) or float(r["pitch"]) >= 30:
            continue
        bag_vals[r["bag"], r["method"]].append(r)
    ws = wb.create_sheet("Boden+Wand")
    cols = [("floor_abs_rel_0_1", "Boden 0–1 m"), ("floor_abs_rel_1_3", "Boden 1–3 m"),
            ("floor_abs_rel_3_6", "Boden 3–6 m"), ("wall_abs_rel", "Wand 4,87 m")]
    ws.append(["Mittlerer relativer Fehler, Kopf flach, Szenen ohne Roboter"] + [c[1] for c in cols])
    for m, name in [("ZED", "ZED Stereo"), ("Metric-Large roh", "DA3 Metric-Large roh"),
                    ("Base skaliert", "DA3 Base skaliert"), ("Metric-Large skaliert", "DA3 Metric-Large skaliert")]:
        cells = []
        for key, _ in cols:
            per_bag = []
            for (bag, mm), rs in bag_vals.items():
                vals = [float(r[key]) for r in rs if mm == m and r[key] not in ("", "nan")]
                if vals:
                    per_bag.append(median(vals))
            cells.append(median(per_bag) if per_bag else None)
        ws.append([name] + cells)
    style(ws, [40, 14, 14, 14, 14], percent_cols="BCDE")
    color(ws, "BCDE", (2, 5))
    ws.append([])
    ws.append(["Boden-Referenz ist die Ebene aus der Kamerapose (TF), selbst auf 0,1–0,5° unsicher. "
               "Wand gegen Maßband. ZED an der Wand im PERFORMANCE-Modus mit Löchern (nur 65 % gültig)."])

    # Head pitch: shift for +1 degree error, parsed from pitch_sensitivity.txt
    ws = wb.create_sheet("Kopfneigung")
    ws.append(["Ball", "IPM (Bit-Bots) Verschiebung (cm)", "DA3 Base skaliert Verschiebung (cm)",
               "DA3 Metric-Large skaliert Verschiebung (cm)"])
    text = (RES / "pitch_sensitivity.txt").read_text()
    for block in text.split("\n\n"):
        head = block.strip().splitlines()
        if not head or not head[0].startswith("Ball"):
            continue
        hdr = head[1].split()
        i = hdr.index("+1.0°") - 3  # value columns start after the three words "Fehler in TF"
        vals = {}
        for line in head[2:]:
            m = re.match(r"(.+?)\s+([+-]\d.*)$", line)
            vals[m.group(1).strip()] = abs(float(m.group(2).split()[i]))
        ws.append([head[0].split(" (")[0], vals.get("IPM"), vals.get("Base skaliert"), vals.get("Metric-Large skaliert")])
    style(ws, [20, 18, 18, 20])
    ws.append([])
    ws.append(["Um wie viele cm wandert der geschätzte Ballabstand, wenn die Kopfneigung aus TF um 1° falsch ist. "
               "Rechenversuch auf echten Bildern (Neigung künstlich verstellt), noch nicht mit echten Kopfhaltungen geprüft."])

    out = RES / "Messungen.xlsx"
    wb.save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
