import re
import unicodedata

def clean_text(text: str) -> str:
    """
    Normalize text by removing extra whitespace and normalizing unicode.
    
    Args:
        text: Input string.
    
    Returns:
        Cleaned string.
    """
    if not isinstance(text, str):
        return ""
    
    # Normalize unicode characters to NFKC form (compatibility decomposition)
    text = unicodedata.normalize('NFKC', text)
    
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
