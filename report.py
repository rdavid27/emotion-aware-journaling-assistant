from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import pandas as pd

def generate_pdf():
    doc = SimpleDocTemplate("emotion_report.pdf")
    styles = getSampleStyleSheet()

    elements = []

    df = pd.read_csv("journal_log.csv")

    elements.append(Paragraph("Emotion Report", styles["Title"]))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph(f"Total Entries: {len(df)}", styles["Normal"]))
    elements.append(Spacer(1, 10))

    most_common = df["emotion"].value_counts().idxmax()
    elements.append(Paragraph(f"Most Common Emotion: {most_common}", styles["Normal"]))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("Recent Entries:", styles["Heading2"]))
    elements.append(Spacer(1, 10))

    for _, row in df.tail(10).iterrows():
        text = f"{row['date']} - {row['emotion']} - {row['text']}"
        elements.append(Paragraph(text, styles["Normal"]))
        elements.append(Spacer(1, 10))

    doc.build(elements)