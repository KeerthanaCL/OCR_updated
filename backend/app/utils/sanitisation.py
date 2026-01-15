import re
import random
import calendar

def random_mm_dd(year: int) -> str:
    """Generate a valid random MM-DD for a given year."""
    month = random.randint(1, 12)
    max_day = calendar.monthrange(year, month)[1]
    day = random.randint(1, max_day)
    return f"{month:02d}-{day:02d}"

def normalize_key(key: str) -> str:
    """
    Normalize OCR keys to a comparable form without hardcoding schemas.
    """
    key = key.lower().strip()
    key = re.sub(r'[^a-z0-9\s]', '', key)
    key = re.sub(r'\s+', ' ', key)
    return key

def should_mask_identifier(normalized_key: str) -> bool:
    """
    Decide whether a key represents a sensitive identifier
    without assuming digit length.
    """
    keywords = [
        "member id",
        "member number",
        "member no",
        "subscriber id",
        "reference number",
        "reference no",
        "ref number",
        "policy number",
        "policy no",
        "claim number",
        "claim id"
    ]

    return any(k in normalized_key for k in keywords)

def sanitize_sensitive_info(text: str) -> str:
    if not text:
        return text

    # --- DOB patterns ---
    dob_pattern1 = re.compile(
        r'\b(?P<m>\d{1,2})[/-](?P<d>\d{1,2})[/-](?P<y>\d{4})\b'
    )
    dob_pattern2 = re.compile(
        r'\b(?P<y>\d{4})[/-](?P<m>\d{1,2})[/-](?P<d>\d{1,2})\b'
    )
    dob_pattern3 = re.compile(
        r'\b(?P<month_name>January|February|March|April|May|June|July|August|September|October|November|December)\s+'
        r'(?P<d>\d{1,2}),?\s+(?P<y>\d{4})\b',
        re.IGNORECASE
    )

    # --- SSN ---
    ssn_pattern = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')

    # --- Generic key-value line ---
    kv_pattern = re.compile(
        r'^\s*(?P<key>[A-Za-z][A-Za-z\s/_-]{1,60})\s*[:\-]\s*(?P<value>.+?)\s*$'
    )

    # --- DOB replacer ---
    def replace_dob(match):
        year = int(match.group("y"))
        return f"{random_mm_dd(year)}-{year}"

    # --- First pass: global replacements ---
    text = dob_pattern1.sub(replace_dob, text)
    text = dob_pattern2.sub(replace_dob, text)
    text = dob_pattern3.sub(replace_dob, text)
    text = ssn_pattern.sub("***-**-****", text)

    # --- Second pass: line-by-line KV sanitization ---
    sanitized_lines = []

    for line in text.splitlines():
        match = kv_pattern.match(line)

        if not match:
            sanitized_lines.append(line)
            continue

        key = match.group("key").strip()
        value = match.group("value").strip()

        normalized_key = normalize_key(key)

        if should_mask_identifier(normalized_key):
            sanitized_value = "<REDACTED>"
        else:
            sanitized_value = value

        sanitized_lines.append(f"{key}: {sanitized_value}")

    return "\n".join(sanitized_lines)
