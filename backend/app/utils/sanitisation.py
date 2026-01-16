import re
import random
import calendar

# =================================================
# DATE HELPERS
# =================================================

def random_mm_dd(year: int) -> str:
    """Generate a valid random MM-DD for a given year."""
    month = random.randint(1, 12)
    max_day = calendar.monthrange(year, month)[1]
    day = random.randint(1, max_day)
    return f"{month:02d}-{day:02d}"

# =================================================
# OCR NORMALIZATION HELPERS
# =================================================

def normalize_ocr_characters(text: str) -> str:
    """
    Fix common OCR character confusions.
    """
    replacements = {
        "|": "I",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "—": "-",
        "–": "-",
    }

    for src, tgt in replacements.items():
        text = text.replace(src, tgt)

    return text


def normalize_number_word(text: str) -> str:
    """
    Normalize OCR variations of 'number / no'.
    """
    # no / no. / n0 / n° → no
    text = re.sub(r'\b(n[o0]|n°|no\.)\b', 'no', text, flags=re.IGNORECASE)

    # number / numbcr → number
    text = re.sub(r'\bnumb[ec]r\b', 'number', text, flags=re.IGNORECASE)

    text = re.sub(r'\b#\b', 'number', text)

    return text


def normalize_line_breaks(text: str) -> str:
    """
    Preserve paragraph structure.
    ONLY fix hyphenated line breaks.
    """
    # Fix hyphenated words split across lines
    text = re.sub(r'-\n(?=\w)', '', text)

    # Normalize excessive blank lines (but keep paragraphs)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text

def redact_signature(text: str) -> str:
    """
    Redact signer name following a closing like 'Sincerely'.
    """
    lines = text.splitlines()
    result = []

    redact_next = False

    for line in lines:
        stripped = line.strip()

        if redact_next and stripped:
            result.append("<REDACTED>")
            redact_next = False
            continue

        result.append(line)

        if re.match(
            r'^\s*(Sincerely|Regards|Respectfully|Yours truly),?\s*$',
            stripped,
            re.IGNORECASE
        ):
            redact_next = True

    return "\n".join(result)


def fix_common_ocr_words(text: str) -> str:
    """
    Targeted fixes for common OCR misspellings.
    """
    fixes = {
        r'\bgrievence\b': 'grievance',
        r'\bappeai\b': 'appeal',
        r'\bcIaim\b': 'claim',
        r'\bpol1cy\b': 'policy',
    }

    for pattern, replacement in fixes.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text

# =================================================
# KEY NORMALIZATION & MASKING LOGIC
# =================================================

def normalize_key(key: str) -> str:
    """
    Normalize OCR keys to a comparable form.
    """
    key = key.lower().strip()
    key = re.sub(r'[^a-z0-9\s]', '', key)
    key = re.sub(r'\s+', ' ', key)
    return key


def should_mask_identifier(normalized_key: str) -> bool:
    keywords = [
        "patient name",
        "patient id",
        "patient number",
        "patient no",
        "member id",
        "member number",
        "member no",
        "subscriber id",
        "reference number",
        "reference no",
        "policy number",
        "policy no",
        "claim number",
        "claim no",
        "claim id",
        "from",
    ]

    # Explicitly allow "patient" ONLY as a KV key
    if normalized_key == "patient":
        return True

    return any(k in normalized_key for k in keywords)

# =================================================
# MAIN PUBLIC FUNCTION
# =================================================

def sanitize_sensitive_info(text: str) -> str:
    """
    Single public sanitization call.
    Performs OCR cleanup + PII masking + date randomization
    + safe signature redaction.
    """
    if not text:
        return text

    # -----------------------------
    # OCR CLEANUP (SAFE ONLY)
    # -----------------------------
    text = normalize_ocr_characters(text)
    text = normalize_number_word(text)
    text = fix_common_ocr_words(text)

    # Preserve paragraph structure – only fix hyphenated breaks
    text = re.sub(r'-\n(?=\w)', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # -----------------------------
    # DATE PATTERNS
    # -----------------------------
    date_pattern1 = re.compile(
        r'\b(?P<m>\d{1,2})[/-](?P<d>\d{1,2})[/-](?P<y>\d{4})\b'
    )
    date_pattern2 = re.compile(
        r'\b(?P<y>\d{4})[/-](?P<m>\d{1,2})[/-](?P<d>\d{1,2})\b'
    )
    date_pattern3 = re.compile(
        r'\b(?P<month_name>January|February|March|April|May|June|July|August|September|October|November|December)\s+'
        r'(?P<d>\d{1,2}),?\s+(?P<y>\d{4})\b',
        re.IGNORECASE
    )

    ssn_pattern = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')

    kv_pattern = re.compile(
        r'^\s*(?P<key>[A-Za-z][A-Za-z\s/_-]{1,60})\s*[:\-]\s*(?P<value>.+?)\s*$'
    )

    def replace_date(match):
        year = int(match.group("y"))
        return f"{random_mm_dd(year)}-{year}"

    # -----------------------------
    # GLOBAL SANITIZATION
    # -----------------------------
    text = date_pattern1.sub(replace_date, text)
    text = date_pattern2.sub(replace_date, text)
    text = date_pattern3.sub(replace_date, text)
    text = ssn_pattern.sub("***-**-****", text)

    # -----------------------------
    # KEY–VALUE MASKING
    # -----------------------------
    sanitized_lines = []

    for line in text.splitlines():
        match = kv_pattern.match(line)

        if not match:
            sanitized_lines.append(line)
            continue

        key = match.group("key").strip()
        value = match.group("value").strip()

        if should_mask_identifier(normalize_key(key)):
            sanitized_lines.append(f"{key}: <REDACTED>")
        else:
            sanitized_lines.append(f"{key}: {value}")

    text = "\n".join(sanitized_lines)

    # -----------------------------
    # SIGNATURE REDACTION (FINAL STEP)
    # -----------------------------
    text = redact_signature(text)

    return text
