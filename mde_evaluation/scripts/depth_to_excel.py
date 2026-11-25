"""One Excel workbook per recording (bag) of 04.11.: the camera image, the measured distances and the
depth maps of ZED and all DA3 models, pixel by pixel.

For every bag the first frame with ZED depth is used (all frames of a bag show the same scene).
Frames listed in EXTRA_FRAMES get an extra file <bag>_bild<N>.xlsx (where things move between frames).
Sheets:
  Messung           camera image with the marked objects, and per object: tape distance and the estimate
                    of every method with its error (taken from results/real_26-11-04/objects.csv)
  ZED               ZED stereo depth in metres
  DA3 Small skaliert         DA3 Small, scaled with the floor 0.5-2.5 m in front of the robot (as in eval_real.py)
  DA3 Base skaliert          DA3 Base, scaled the same way
  DA3 Metric-Large skaliert  DA3 Metric-Large, scaled the same way
  DA3 Metric-Large roh       DA3 Metric-Large as the model outputs it (focal length * output / 300)
Depth values are z-depth (along the camera axis) in metres, one cell per pixel; empty = no value.
All depth sheets share one colour scale (near blue, far red) so they can be compared by eye.

Run eval_real.py first (it fills data/da3/ and writes objects.csv).
Usage: python scripts/depth_to_excel.py   -> results/tiefe_excel_26-11-04/<bag>.xlsx
"""

import csv
import io
import sys
from pathlib import Path

import cv2
import numpy as np
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

sys.path.insert(0, str(Path(__file__).parent))
from eval_real import OUT as RESULTS  # noqa: E402
from eval_real import ROOT, add_da3, depth_variants, load_frames  # noqa: E402

OUT = ROOT / "results/tiefe_excel_26-11-04"
NEAR, FAR = 0.3, 6.0  # colour scale in metres, the same for every sheet
SHEETS = [("ZED", "ZED"), ("DA3 Small skaliert", "Small skaliert"), ("DA3 Base skaliert", "Base skaliert"),
          ("DA3 Metric-Large skaliert", "Metric-Large skaliert"), ("DA3 Metric-Large roh", "Metric-Large roh")]
TABLE_METHODS = ["IPM", "ZED", "Small skaliert", "Base skaliert", "Metric-Large skaliert", "Metric-Large roh"]
# Extra frames that get their own file (<bag>_bild<N>.xlsx): in 'mensch' a person moves the ball,
# frame 3 shows it held in the air
EXTRA_FRAMES = {"mde_s1_performance_mensch": (1, 2, 3, 7)}
NOTE = {"mostly hidden": " (fast verdeckt)", "half hidden": " (halb verdeckt)", "in the air": " (in der Luft)"}
TABLE_NAMES = ["IPM (Bit-Bots)", "ZED", "DA3 Small skaliert", "DA3 Base skaliert", "DA3 Metric-Large skaliert",
               "DA3 Metric-Large roh"]


def depth_sheet(wb, title, depth):
    """Depth grid laid out like the image (as in npy_to_text.py), row 1 = pixel columns, column A = pixel rows."""
    ws = wb.create_sheet(title)
    h, w = depth.shape
    ws.cell(1, 1, "Zeile \\ Spalte").font = Font(bold=True)
    for col in range(w):
        ws.cell(1, col + 2, col).font = Font(bold=True)
    rounded = np.round(depth, 3)
    for row in range(h):
        ws.cell(row + 2, 1, row).font = Font(bold=True)
        for col in range(w):
            v = rounded[row, col]
            if np.isfinite(v) and v > 0:
                ws.cell(row + 2, col + 2, float(v)).number_format = "0.000"
    ws.freeze_panes = "B2"
    for col in range(2, w + 2):
        ws.column_dimensions[get_column_letter(col)].width = 6
    ws.conditional_formatting.add(
        f"B2:{get_column_letter(w + 1)}{h + 1}",
        ColorScaleRule(start_type="num", start_value=NEAR, start_color="3B4CC0", mid_type="num",
                       mid_value=(NEAR + FAR) / 2, mid_color="F7F7F7", end_type="num", end_value=FAR,
                       end_color="B40426"))


def marked_image(d, only=None):
    """Camera image with the outline and label of every marked object (or only object `only`), as PNG bytes."""
    img = cv2.cvtColor(d["rgb"], cv2.COLOR_RGB2BGR).copy()
    for i_obj, obj, core, _shape in d["objs"]:
        if only is not None and i_obj != only:
            continue
        contours, _ = cv2.findContours(core.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 255, 255), 2)
        if contours:
            x, y, _w, _h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            cv2.putText(img, str(i_obj + 1), (x, max(y - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    ok, png = cv2.imencode(".png", img)
    return io.BytesIO(png.tobytes())


def measurement_sheet(ws, d, rows):
    ws.title = "Messung"
    ws["A1"] = f"Messung {d['bag'].removeprefix('mde_s1_')}, Bild {d['file']} (Aufnahme 04.11.)"
    ws["A1"].font = Font(bold=True, size=13)
    ws["A2"] = ("Abstände waagerecht in m. Wahr ab Robotermitte = Maßband (ab Fußspitze) + 9,9 cm, gezählt ab "
                "base_footprint wie im IPM der Bit-Bots. Fehler in %: + = zu weit, "
                "− = zu nah. Die Tiefenblätter zeigen z-Tiefe pro Pixel (Farbskala 0,3 m blau bis 6 m rot).")
    header = ["Nr.", "Objekt", "Maßband ab Fußspitze", "Wahr ab Robotermitte"] + [n for name in TABLE_NAMES for n in (name, "Fehler")]
    for c, v in enumerate(header, start=1):
        ws.cell(4, c, v).font = Font(bold=True)
    r = 5
    for i_obj, obj, *_ in d["objs"]:
        est = {row["method"]: row for row in rows if int(row["object"]) == i_obj}
        if not est:
            continue
        first = next(iter(est.values()))
        truth = float(first["truth"])
        kind = "Ball" if first["kind"] == "ball" else "Roboter"
        cells = [i_obj + 1, kind + NOTE.get(first["note"], ""), float(first["tape"]), round(truth, 2)]
        for m in TABLE_METHODS:
            e = float(est[m]["estimate"]) if m in est and est[m]["estimate"] not in ("", "nan") else None
            cells += [round(e, 2) if e is not None else None, (e - truth) / truth if e is not None else None]
        for c, v in enumerate(cells, start=1):
            cell = ws.cell(r, c, v)
            if header[c - 1] == "Fehler":
                cell.number_format = "+0.0%;-0.0%;0.0%"
        r += 1
    if r == 5:
        ws.cell(5, 2, "in diesem Bild ist kein Objekt markiert")
        r = 6
    ws.column_dimensions["B"].width = 30
    img = XLImage(marked_image(d))
    ws.add_image(img, f"B{r + 2}")


def excel_name(bag, idx=None):
    """<bag>.xlsx for the first frame of a bag, <bag>_bild<N>.xlsx for an extra frame N."""
    short = bag.removeprefix("mde_s1_")
    return f"{short}.xlsx" if idx is None else f"{short}_bild{idx}.xlsx"


def main():
    data = load_frames()
    add_da3(data)
    objects = list(csv.DictReader(open(RESULTS / "objects.csv")))
    OUT.mkdir(parents=True, exist_ok=True)
    seen = set()
    for d in data:
        extra = d["idx"] in EXTRA_FRAMES.get(d["bag"], ())
        if d["bag"] in seen and not extra:
            continue
        seen.add(d["bag"])
        variants = depth_variants(d)
        wb = Workbook()
        rows = [o for o in objects if o["bag"] == d["bag"] and o["file"] == d["file"]]
        measurement_sheet(wb.active, d, rows)
        for title, key in SHEETS:
            depth_sheet(wb, title, variants[key])
        out = OUT / excel_name(d["bag"], d["idx"] if extra else None)
        wb.save(out)
        print("saved", out)


if __name__ == "__main__":
    main()
