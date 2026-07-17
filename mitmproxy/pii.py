"""PII / secret detection + redaction via Presidio, used on BOTH the request
(before prompts leave for a third-party model) and the response. Adds a custom
recognizer for API keys / bearer tokens on top of Presidio's built-ins."""
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine

_SECRET_PATTERNS = [
    Pattern("openai_key", r"sk-[A-Za-z0-9]{20,}", 0.9),
    Pattern("aws_key", r"AKIA[0-9A-Z]{16}", 0.9),
    Pattern("bearer", r"(?i)bearer\s+[A-Za-z0-9._\-]{20,}", 0.6),
]
_ENTITIES = ["CREDIT_CARD", "US_SSN", "EMAIL_ADDRESS",
             "PHONE_NUMBER", "IP_ADDRESS", "API_SECRET"]

_analyzer = AnalyzerEngine()
_analyzer.registry.add_recognizer(
    PatternRecognizer(supported_entity="API_SECRET", patterns=_SECRET_PATTERNS)
)
_anonymizer = AnonymizerEngine()

def scan_and_redact(text: str, language: str = "en"):
    """Return (redacted_text, [entity_types_found]). Empty list => nothing found."""
    results = _analyzer.analyze(text=text, entities=_ENTITIES, language=language)
    if not results:
        return text, []
    redacted = _anonymizer.anonymize(text=text, analyzer_results=results).text
    return redacted, sorted({r.entity_type for r in results})
