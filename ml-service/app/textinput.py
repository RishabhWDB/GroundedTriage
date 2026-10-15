"""The single definition of the text the classifiers see.

Used to build the `text` column of the training splits (data.build) and by the
ml-service at inference, so the fine-tuned models on the PC and the serving
path format input identically. Changing it requires re-exporting splits and
retraining.
"""

MODEL_INPUT_FORMAT = "subject + blank line + body (body only when subject is empty)"


def model_input(subject: str | None, body: str | None) -> str:
    subject, body = (subject or "").strip(), (body or "").strip()
    return f"{subject}\n\n{body}" if subject else body
