from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from datetime import datetime


def generate_legal_pdf_report(output_path, metadata, candidates, segments, conclusion):

    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'title',
        fontSize=20,
        leading=24,
        alignment=1,
        spaceAfter=20,
        textColor=colors.HexColor("#002147")
    )

    header_style = ParagraphStyle(
        'header',
        fontSize=14,
        leading=16,
        spaceAfter=10,
        textColor=colors.HexColor("#00C853"),
        underlineWidth=1
    )

    normal = styles["Normal"]

    content = []

    # ---------------------
    # TITRE
    # ---------------------
    content.append(Paragraph("Rapport Officiel d’Analyse de Plagiat", title_style))
    content.append(Spacer(1, 12))

    # ---------------------
    # MÉTADONNÉES
    # ---------------------
    content.append(Paragraph("<b>Informations générales</b>", header_style))

    meta_table_data = [
        ["ID Analyse", metadata.get("uid")],
        ["Fichier analysé", metadata.get("fichier")],
        ["SimHash", str(metadata.get("simhash"))],
        ["Date", metadata.get("date")],
    ]

    meta_table = Table(meta_table_data, colWidths=[5*cm, 10*cm])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.HexColor("#eeeeee")),
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')
    ]))
    content.append(meta_table)
    content.append(Spacer(1, 20))

    # ---------------------
    # TOP 5 CANDIDATES
    # ---------------------
    content.append(Paragraph("<b>Top 5 des similarités détectées</b>", header_style))

    if not candidates:
        content.append(Paragraph("Aucune similarité significative n’a été détectée.", normal))
    else:
        table_data = [["ID Œuvre", "Titre", "Score Similarité", "Segments correspondants"]]

        for c in candidates[:5]:
            table_data.append([
                c.get("oeuvre_id"),
                c.get("titre"),
                f"{c.get('score'):.3f}",
                str(c.get("match_count"))
            ])

        table = Table(table_data, colWidths=[4*cm, 5*cm, 3*cm, 4*cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#dddddd")),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')
        ]))
        content.append(table)

    content.append(Spacer(1, 20))

    # ---------------------
    # SEGMENTS > 70%
    # ---------------------
    content.append(Paragraph("<b>Extraits présentant plus de 70% de similarité</b>", header_style))

    if not segments:
        content.append(Paragraph("Aucun segment significatif n’a été identifié.", normal))
    else:
        for seg in segments:
            content.append(Paragraph(seg.replace("\n", "<br/>"), normal))
            content.append(Spacer(1, 10))

    content.append(Spacer(1, 20))

    # ---------------------
    # CONCLUSION JURIDIQUE
    # ---------------------
    content.append(Paragraph("<b>Conclusion de l’expertise</b>", header_style))
    content.append(Paragraph(conclusion.replace("\n", "<br/>"), normal))

    doc.build(content)
