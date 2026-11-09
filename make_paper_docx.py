"""Render the submitted Markdown paper as a clean Word document."""
from pathlib import Path
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

root = Path(__file__).resolve().parent
doc = Document()
section = doc.sections[0]
section.top_margin = section.bottom_margin = Inches(0.85)
section.left_margin = section.right_margin = Inches(0.9)
style = doc.styles['Normal']
style.font.name = 'Times New Roman'
style.font.size = Pt(11)
style.paragraph_format.space_after = Pt(6)

lines = (root / 'research_paper.md').read_text(encoding='utf-8').splitlines()
index = 0
while index < len(lines):
    raw = lines[index]
    line = raw.strip()
    if line.startswith('|'):
        table_lines = []
        while index < len(lines) and lines[index].strip().startswith('|'):
            candidate = lines[index].strip()
            if not set(candidate.replace('|', '').replace('-', '').replace(':', '').replace(' ', '')):
                index += 1
                continue
            table_lines.append([cell.strip() for cell in candidate.strip('|').split('|')])
            index += 1
        if table_lines:
            table = doc.add_table(rows=0, cols=len(table_lines[0]))
            table.style = 'Table Grid'
            for row_values in table_lines:
                cells = table.add_row().cells
                for cell, value in zip(cells, row_values):
                    cell.text = value
        continue
    if not line:
        index += 1
        continue
    if line.startswith('# '):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(line[2:])
        run.bold = True
        run.font.size = Pt(16)
    elif line.startswith('## '):
        doc.add_heading(line[3:], level=1)
    elif line.startswith('### '):
        doc.add_heading(line[4:], level=2)
    elif line.startswith('**') and line.endswith('**'):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run(line[2:-2]).bold = True
    elif line.startswith('* '):
        doc.add_paragraph(line[2:], style='List Bullet')
    elif line[:2].isdigit() and line[1:3] == '. ':
        doc.add_paragraph(line[3:], style='List Number')
    else:
        doc.add_paragraph(line.replace('**', '').replace('`', ''))
    index += 1

doc.add_page_break()
doc.add_paragraph('Appendix: exact measured results', style='Heading 1')
doc.add_paragraph((root / 'results' / 'enhanced_metrics.json').read_text(encoding='utf-8'))
doc.save(root / 'research_paper.docx')
print(root / 'research_paper.docx')
