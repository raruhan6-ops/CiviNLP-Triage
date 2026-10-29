import re

# Emails
EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')

# US phone numbers. A separator, parentheses or a +1 prefix is required, so bare
# 10-digit IDs (e.g. NYC BBL tax-lot numbers) are left alone.
PHONE_REGEX = re.compile(
    r'(?:\+1[-.\s]?)?(?:\(\d{3}\)\s?|\b\d{3}[-.\s])\d{3}[-.\s]\d{4}\b'
    r'|\+1\d{10}\b'
)

# Street addresses: house number + 1-4 words + a street suffix ("123 W 45th St")
_STREET_SUFFIXES = (
    "street|st|avenue|ave|road|rd|boulevard|blvd|place|pl|drive|dr|lane|ln|"
    "court|ct|parkway|pkwy|terrace|ter|highway|hwy|way|broadway"
)
ADDRESS_REGEX = re.compile(
    rf"\b\d{{1,6}}(?:-\d{{1,4}})?\s+(?:[A-Za-z0-9.']+\s+){{0,4}}(?:{_STREET_SUFFIXES})\b\.?",
    re.IGNORECASE,
)


def redact_pii(text: str) -> str:
    """
    Redact potential PII (emails, phone numbers, street addresses) from text.

    Names are not detected; the public NYC 311 fields used as model input
    (complaint_type, descriptor) contain no free text, but any free-text source
    added later needs a proper NER-based redactor.

    Args:
        text: Input string.

    Returns:
        Text with PII replaced by [EMAIL_REDACTED], [PHONE_REDACTED], [ADDRESS_REDACTED].
    """
    if not isinstance(text, str):
        return ""

    text = EMAIL_REGEX.sub('[EMAIL_REDACTED]', text)
    text = PHONE_REGEX.sub('[PHONE_REDACTED]', text)
    text = ADDRESS_REGEX.sub('[ADDRESS_REDACTED]', text)
    return text
