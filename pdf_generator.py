# pdf_generator.py
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
import os
import re
from pathlib import Path

def generate_pdf_report(markdown_content, output_path="root_cause_report.pdf"):
    """Converte il report Markdown in un PDF elegante usando ReportLab."""
    # Assicura che la directory esista
    output_dir = Path(output_path).parent
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Setup documento
    doc = SimpleDocTemplate(output_path, pagesize=A4, 
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=72)
    
    # Elementi da aggiungere al PDF
    elements = []
    
    # Stili
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Title', fontSize=18, alignment=1, spaceAfter=12, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name='Heading2', fontSize=16, spaceAfter=10, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name='BodyText', fontSize=12, leading=14, spaceAfter=8))
    styles.add(ParagraphStyle(name='Small', fontSize=10, leading=12))
    
    # Logo se disponibile
    logo_path = os.path.join("assets", "logo.png")
    if os.path.exists(logo_path):
        img = Image(logo_path, width=120, height=60)
        elements.append(img)
    
    # Estrai sezioni dal markdown
    # Titolo
    title_match = re.search(r'# (.+?)\n', markdown_content)
    if title_match:
        title = title_match.group(1)
        elements.append(Paragraph(title, styles['Title']))
    else:
        elements.append(Paragraph("Root-Cause Discovery Report", styles['Title']))
    
    # Spaziatura
    elements.append(Spacer(1, 12))
    
    # Estrai tabella dei driver dal markdown
    table_match = re.search(r'\| Rank \| Driver \|.*\n\|.*\n((?:\|.*\n)+)', markdown_content)
    if table_match:
        table_content = table_match.group(1).strip()
        rows = []
        # Headers
        headers = ["Rank", "Driver", "Description", "Effect Size", "p-value", 
                  "Business Validation", "Strength", "Business Context"]
        rows.append(headers)
        
        # Parsing delle righe della tabella
        for line in table_content.split('\n'):
            if line.startswith('|'):
                # Rimuovi i bordi | e dividi per |
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                if len(cells) >= 7:  # Assicurati che ci siano abbastanza celle
                    rows.append(cells)
        
        # Crea la tabella
        if len(rows) > 1:
            elements.append(Paragraph("Validated Top Drivers", styles['Heading2']))
            elements.append(Spacer(1, 6))
            
            table = Table(rows, repeatRows=1)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            elements.append(table)
    
    # Outlier Check
    outlier_match = re.search(r'## Outlier Check\n\n(.+?)(?:\n\n|$)', markdown_content, re.DOTALL)
    if outlier_match:
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Outlier Check", styles['Heading2']))
        outlier_text = outlier_match.group(1).strip()
        elements.append(Paragraph(outlier_text, styles['BodyText']))
    
    # Normal Operating Ranges
    ranges_match = re.search(r'## Normal Operating Ranges\n\n((?:- .+\n)+)', markdown_content)
    if ranges_match:
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Normal Operating Ranges", styles['Heading2']))
        
        ranges_text = ranges_match.group(1).strip()
        for range_line in ranges_text.split('\n'):
            if range_line.startswith('- '):
                range_content = range_line[2:].strip()
                elements.append(Paragraph("• " + range_content, styles['BodyText']))
    
    # User Notes
    notes_match = re.search(r'## User Notes\n\n(.+?)(?:\n\n|$)', markdown_content, re.DOTALL)
    if notes_match:
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("User Notes", styles['Heading2']))
        notes_text = notes_match.group(1).strip()
        elements.append(Paragraph(notes_text, styles['BodyText']))
    
    # Footer
    elements.append(Spacer(1, 30))
    elements.append(Paragraph(f"Report generato il {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles['Small']))
    elements.append(Paragraph("© 2025 Crossnection", styles['Small']))
    
    # Genera il PDF
    doc.build(elements)
    return output_path