"""Render research_paper.md as an IEEE-style two-column Word manuscript."""
from pathlib import Path
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parent


def set_columns(section, columns=2, space=0.22):
    cols = section._sectPr.xpath('./w:cols')[0]
    cols.set(qn('w:num'), str(columns))
    cols.set(qn('w:space'), str(int(space * 1440)))


def set_cell_text(cell, value, bold=False):
    cell.text = ''
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(value)
    run.bold = bold
    run.font.name = 'Times New Roman'
    run.font.size = Pt(7.5)


def add_body_paragraph(doc, text, first_line=True):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.0
    if first_line:
        p.paragraph_format.first_line_indent = Inches(0.14)
    p.add_run(text.replace('**', '').replace('`', ''))
    return p


doc = Document()
section = doc.sections[0]
section.top_margin = Inches(0.75)
section.bottom_margin = Inches(0.75)
section.left_margin = Inches(0.625)
section.right_margin = Inches(0.625)
set_columns(section, columns=1)
normal = doc.styles['Normal']
normal.font.name = 'Times New Roman'
normal.font.size = Pt(10)
normal.paragraph_format.space_after = Pt(0)
for name in ['Heading 1', 'Heading 2']:
    doc.styles[name].font.name = 'Times New Roman'
    doc.styles[name].font.bold = True
    doc.styles[name].font.size = Pt(10)

lines = (ROOT / 'research_paper.md').read_text(encoding='utf-8').splitlines()
title = lines[0].replace('# ', '').strip()
subtitle = lines[2].replace('## ', '').strip()
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after = Pt(5)
r = p.add_run(title); r.font.name = 'Times New Roman'; r.font.size = Pt(24)
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run(subtitle); r.italic = True; r.font.name = 'Times New Roman'; r.font.size = Pt(11)
for author_line in ['K Prem Sai', 'Registration No. 11249A148', 'NLP Mini Project']:
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after = Pt(0)
    r = p.add_run(author_line); r.font.name = 'Times New Roman'; r.font.size = Pt(11 if author_line == 'K Prem Sai' else 9)

body = doc.add_section(WD_SECTION.CONTINUOUS)
body.top_margin = section.top_margin; body.bottom_margin = section.bottom_margin
body.left_margin = section.left_margin; body.right_margin = section.right_margin
set_columns(body, columns=2)

index = next(i for i, value in enumerate(lines) if value.strip() == '## Abstract') + 1
in_abstract = True
while index < len(lines):
    line = lines[index].strip()
    if not line:
        index += 1; continue
    if line.startswith('|'):
        rows = []
        while index < len(lines) and lines[index].strip().startswith('|'):
            candidate = lines[index].strip(); index += 1
            if not set(candidate.replace('|', '').replace('-', '').replace(':', '').replace(' ', '')):
                continue
            rows.append([cell.strip() for cell in candidate.strip('|').split('|')])
        if rows:
            table = doc.add_table(rows=0, cols=len(rows[0])); table.style = 'Table Grid'
            for row_number, values in enumerate(rows):
                cells = table.add_row().cells
                for cell, value in zip(cells, values): set_cell_text(cell, value, bold=row_number == 0)
        continue
    if line.startswith('## '):
        heading = line[3:]
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(5); p.paragraph_format.space_after = Pt(2)
        p.add_run('REFERENCES' if heading == 'References' else heading.upper()).bold = True
        in_abstract = False; index += 1; continue
    if line.startswith('### '):
        p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(3); p.paragraph_format.space_after = Pt(1)
        p.add_run(line[4:]).bold = True; index += 1; continue
    if line.startswith('**Keywords:**'):
        p = add_body_paragraph(doc, line, first_line=False); p.runs[0].italic = True; index += 1; continue
    if line.startswith('* '):
        p = add_body_paragraph(doc, line[2:], first_line=False); p.paragraph_format.left_indent = Inches(0.12); index += 1; continue
    if line[:2].isdigit() and line[1:3] == '. ':
        add_body_paragraph(doc, line, first_line=False); index += 1; continue
    if in_abstract:
        p = add_body_paragraph(doc, line, first_line=False); p.runs[0].text = 'Abstract—' + p.runs[0].text; in_abstract = False
    elif line.startswith('Table ') or line.startswith('The baseline model confusion') or line.startswith('The enhanced model confusion'):
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.add_run(line).italic = True
    else:
        add_body_paragraph(doc, line)
    index += 1

doc.save(ROOT / 'research_paper.docx')
print(ROOT / 'research_paper.docx')
