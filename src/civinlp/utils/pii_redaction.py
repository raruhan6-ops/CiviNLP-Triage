import re

# Basic regex for emails and phone numbers
EMAIL_REGEX = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
PHONE_REGEX = r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b'

def redact_pii(text: str) -> str:
    """
    Redact potential PII (emails, phone numbers) from text.
    
    Args:
        text: Input string.
        
    Returns:
        Text with PII replaced by [REDACTED].
    """
    if not isinstance(text, str):
        return ""
    
    # Redact emails
    text = re.sub(EMAIL_REGEX, '[EMAIL_REDACTED]', text)
    
    # Redact phones
    text = re.sub(PHONE_REGEX, '[PHONE_REDACTED]', text)
    
    return text
