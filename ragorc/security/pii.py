"""PII detection and redaction.

Two directions, both necessary:

* **Outbound** — retrieved chunks may carry personal data that should not reach
  a third-party model provider. Redacting before the prompt is built is the only
  point where you still control it.
* **Inbound** — queries and answers get logged and cached; a cached answer
  containing a customer's card number is a durable liability.

Regex detection with validation where the format allows it: a 16-digit string is
only a card number if it passes Luhn, and checking costs nothing. Validation is
what keeps the false-positive rate low enough to run redaction by default.

What this does not do
---------------------
There is no NER-backed detector and no provider switch. :class:`PIIRedactor`
takes ``settings`` and nothing else, and the entity list it walks is
``security.pii_entities`` against the regex table below. The docstring here used
to tell operators to "install Presidio and set ``provider='presidio'``" — a
parameter that has never existed, in a module whose whole job is to be trusted.

The limits that follow are the regex engine's, not a configuration choice:
detection is format-driven, so an entity with no distinguishing shape — a person's
name, a street address, a free-text account reference — is not found at all.
Treat this as a net for the formats it names, not as a guarantee that redacted
text is clean. A deployment needing NER should run the text through its own
detector before handing it to this library.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import structlog

from ragorc.core.settings import SecuritySettings, get_settings

log = structlog.get_logger(__name__)

__all__ = ["PIIFinding", "PIIRedactor"]

_DETECTORS: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(r"\b[\w.!#$%&'*+/=?^`{|}~-]+@[\w-]+(?:\.[\w-]+)+\b"),
    # Deliberately conservative: requires a separator or country code so it does
    # not match every 10-digit identifier in a corpus.
    "PHONE": re.compile(
        r"(?<!\d)(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?|\d{2,4}[\s.-])\d{3,4}[\s.-]?\d{3,4}(?!\d)"
    ),
    "CREDIT_CARD": re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"),
    "SSN": re.compile(r"(?<!\d)(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?!\d)"),
    "IBAN": re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,3})?\b"),
    "IP": re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
    "AWS_KEY": re.compile(r"\b(?:AKIA|ASIA|AROA|AIDA)[0-9A-Z]{16}\b"),
    "PRIVATE_KEY": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"),
    "JWT": re.compile(r"\beyJ[\w-]{10,}\.[\w-]{10,}\.[\w-]{10,}\b"),
    "PASSPORT": re.compile(r"\b[A-Z]{1,2}\d{6,9}\b"),
}

#: Detectors whose format carries a checksum, so a match can be confirmed.
_VALIDATORS = {"CREDIT_CARD": "luhn", "IBAN": "iban_mod97"}


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _iban_mod97(value: str) -> bool:
    compact = re.sub(r"\s+", "", value).upper()
    if len(compact) < 15:
        return False
    rearranged = compact[4:] + compact[:4]
    numeric = "".join(str(ord(c) - 55) if c.isalpha() else c for c in rearranged)
    try:
        return int(numeric) % 97 == 1
    except ValueError:
        return False


def _plausible_phone(text: str, match: re.Match[str]) -> bool:
    """Reject phone matches that are slices of a longer numeric sequence."""
    digits = re.sub(r"\D", "", match.group(0))
    if not 7 <= len(digits) <= 15:  # ITU E.164 range
        return False
    tail = text[match.end() : match.end() + 6]
    if re.match(r"^[\s.-]?\d{3,}", tail):
        return False
    head = text[max(0, match.start() - 6) : match.start()]
    return not re.search(r"\d{3,}[\s.-]?$", head)


@dataclass(slots=True)
class PIIFinding:
    entity: str
    value: str
    start: int
    end: int
    confidence: float = 1.0


@dataclass(slots=True)
class PIIResult:
    text: str
    findings: list[PIIFinding] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return bool(self.findings)

    @property
    def entities(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(f.entity for f in self.findings))


class PIIRedactor:
    """Finds and rewrites personal data."""

    def __init__(self, settings: SecuritySettings | None = None) -> None:
        self.settings = settings or get_settings().security
        self.entities = [e.upper() for e in self.settings.pii_entities]

    def detect(self, text: str) -> list[PIIFinding]:
        if not text:
            return []
        findings: list[PIIFinding] = []
        for entity in self.entities:
            pattern = _DETECTORS.get(entity)
            if pattern is None:
                continue
            validator = _VALIDATORS.get(entity)
            for match in pattern.finditer(text):
                raw = match.group(0)
                confidence = 1.0
                if validator == "luhn":
                    digits = re.sub(r"[ -]", "", raw)
                    if not (13 <= len(digits) <= 19 and _luhn(digits)):
                        continue
                elif validator == "iban_mod97":
                    if not _iban_mod97(raw):
                        continue
                elif entity == "PHONE":
                    # No checksum exists for phone numbers, so reject the two
                    # shapes that produce most false positives: a match that is
                    # really a slice of a longer digit run (an account number,
                    # an unvalidated card), and one with implausible length.
                    if not _plausible_phone(text, match):
                        continue
                    confidence = 0.6
                elif entity in ("PASSPORT", "IP"):
                    # Formats with no checksum: report, but flag the uncertainty
                    # so a caller can choose to only act on high-confidence hits.
                    confidence = 0.6
                findings.append(PIIFinding(entity, raw, match.start(), match.end(), confidence))
        # Overlapping matches (a card number inside a phone pattern): keep the
        # longest, which is the more specific detection.
        findings.sort(key=lambda f: (f.start, -(f.end - f.start)))
        kept: list[PIIFinding] = []
        last_end = -1
        for f in findings:
            if f.start >= last_end:
                kept.append(f)
                last_end = f.end
        return kept

    def redact(self, text: str, *, action: str | None = None) -> PIIResult:
        """Rewrite detected PII. ``hash`` keeps values joinable across records
        (the same email always maps to the same token) without exposing them —
        useful when the corpus needs entity consistency after redaction."""
        if not self.settings.enable_pii_redaction or not text:
            return PIIResult(text=text)
        findings = self.detect(text)
        if not findings:
            return PIIResult(text=text)

        mode = action or self.settings.pii_action
        if mode == "flag":
            return PIIResult(text=text, findings=findings)

        out: list[str] = []
        cursor = 0
        for f in findings:
            out.append(text[cursor : f.start])
            if mode == "hash":
                digest = hashlib.blake2b(f.value.encode(), digest_size=6).hexdigest()
                out.append(f"[{f.entity}:{digest}]")
            else:
                out.append(f"[{f.entity}_REDACTED]")
            cursor = f.end
        out.append(text[cursor:])
        log.info("pii_redacted", entities=[f.entity for f in findings], count=len(findings))
        return PIIResult(text="".join(out), findings=findings)
