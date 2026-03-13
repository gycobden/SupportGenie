import uuid


def create_ticket(title: str, severity: str, summary: str) -> dict:
    """
    Create a structured support ticket.

    Args:
        title: Short title describing the issue.
        severity: Severity level ('low', 'medium', 'high', 'critical').
        summary: Detailed description of the issue.

    Returns:
        A dict containing the ticket_id, title, severity, and summary.
    """
    return {
        "ticket_id": f"T-{uuid.uuid4().hex[:6].upper()}",
        "title": title,
        "severity": severity,
        "summary": summary,
    }
