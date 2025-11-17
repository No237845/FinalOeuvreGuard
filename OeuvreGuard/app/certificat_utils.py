from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.units import cm
import os

def generate_certificat(path, data):
    """
    Génère un certificat PDF propre et bien aligné via ReportLab / Platypus.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    doc = SimpleDocTemplate(
        path,
        pagesize=A4,
        leftMargin=2*cm,
        rightMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )

    styles = getSampleStyleSheet()

    # Styles personnalisés
    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        fontSize=24,
        alignment=TA_CENTER,
        spaceAfter=20
    )

    subtitle_style = ParagraphStyle(
        "SubtitleStyle",
        parent=styles["Normal"],
        fontSize=14,
        alignment=TA_CENTER,
        spaceAfter=30
    )

    field_style = ParagraphStyle(
        "FieldStyle",
        parent=styles["Normal"],
        fontSize=12,
        alignment=TA_LEFT,
        leading=16,
        spaceAfter=6
    )

    message_style = ParagraphStyle(
        "MessageStyle",
        parent=styles["Normal"],
        fontSize=12,
        alignment=TA_LEFT,
        leading=16,
        spaceAfter=15
    )

    signature_style = ParagraphStyle(
        "SignStyle",
        parent=styles["Italic"],
        fontSize=12,
        alignment=TA_LEFT,
        spaceBefore=20
    )

    elements = []

    # Titre
    elements.append(Paragraph("CERTIFICAT D’ENREGISTREMENT D’ŒUVRE", title_style))
    elements.append(Paragraph("Système OeuvreGuard – Protection Numérique", subtitle_style))

    # Données
    fields = [
        ("ID de l'œuvre", data["id"]),
        ("Titre", data["titre"]),
        ("Auteur", data["auteur"]),
        ("Genre", data["genre"]),
        ("Langue", data["langue"]),
        ("Type", data["type"]),
        ("Date d'enregistrement", data["date"]),
        ("Empreinte numérique (hash)", data["hash"])
    ]

    for label, value in fields:
        elements.append(Paragraph(f"<b>{label} :</b> {value}", field_style))

    elements.append(Spacer(1, 20))

    # Texte officiel
    msg = (
        "Ce certificat atteste que l'œuvre ci-dessus a été enregistrée avec succès "
        "dans la plateforme OeuvreGuard. L'auteur est invité à se présenter à la BBDA "
        "munie de ce document pour compléter la procédure d'enregistrement officielle."
    )
    elements.append(Paragraph(msg, message_style))

    # Signature
    elements.append(Paragraph("— Plateforme OeuvreGuard —", signature_style))

    # Génération
    doc.build(elements)
