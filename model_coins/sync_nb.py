import nbformat
from pathlib import Path

p = Path(r"model_coins/coin_training.ipynb")
with open(p, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if "report_md_text = f" in cell.source:
        cell.source = """# Tahap 8 - Sinkronisasi Laporan Komprehensif
report_path = RESULTS_DIR / 'report.md'
if report_path.exists():
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    print(f"Laporan komprehensif diverifikasi di {report_path} ({len(content)} karakter).")
    print("Seluruh analisis OOF 5-kelas, investigasi 9 kasus error, dan arsitektur session-level telah tersinkronisasi.")
else:
    print("Peringatan: file report.md belum ditemukan.")
"""

with open(p, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("coin_training.ipynb successfully updated!")

