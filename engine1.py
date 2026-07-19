from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, ValidationError


INPUT_FILE = Path("mock_emails.json")
OUTPUT_FILE = Path("triage_output.json")
DEFAULT_EXECUTION_METHOD = "Review email text to execute manual task."

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


class ActionItem(BaseModel):
    """A concrete action identified in an email."""
    task: str = Field(min_length=1)
    deadline: Optional[str] = None
    execution_method: str = Field(min_length=1)


class TriageResult(BaseModel):
    """Strict, serializable contract for a triaged email."""
    email_id: str = Field(min_length=1)
    sender: str = Field(min_length=1)
    summary: str
    is_suspicious: bool
    risk_reason: Optional[str] = None
    importance_score: int = Field(ge=1, le=10)
    action_required: bool
    action_items: list[ActionItem]


SPAM_PATTERNS: dict[str, str] = {
    r"\.xyz(?:\b|/)": "Sender or content uses a .xyz domain.",
    r"verify[-\s]identity": "Message asks the recipient to verify identity.",
    r"unauthorized\s+login": "Message reports an unauthorized login.",
    r"\brestricted\b": "Message claims access is restricted.",
    r"action\s+required": "Message uses an action-required prompt.",
}
URGENCY_PATTERN = re.compile(r"\b(?:urgent|critical|deadline|immediately)\b", re.IGNORECASE)

DIRECTIVE_PATTERN = re.compile(
    r"\b(?:can someone|need us to|must log|upgrade your|update your|could you|"
    r"please ensure|please update|please (?:review|send|sign)|remember to)\b",
    re.IGNORECASE,
)

DEADLINE_PATTERNS = (
    re.compile(r"\bby\s+tomorrow(?:\s+(?:morning|afternoon|evening|at\s+[^.,;!?]+))?", re.IGNORECASE),
    re.compile(r"\b(?:by\s+)?(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
               r"(?:\s*,?\s*(?:January|February|March|April|May|June|July|August|September|October|November|December)"
               r"\s+\d{1,2}(?:st|nd|rd|th)?(?:\s+at\s+[^.,;!?]+)?)?", re.IGNORECASE),
    re.compile(r"\b(?:by|before|within)\s+\d+\s+(?:hours?|days?|weeks?)\b", re.IGNORECASE),
    re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
               r"\s+\d{1,2}(?:st|nd|rd|th)?(?:\s+at\s+[^.,;!?]+)?", re.IGNORECASE),
)


def _as_text(value: Any, field_name: str) -> str:
    """Return a required email field as stripped text or raise a useful error."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Email field '{field_name}' must be a non-empty string.")
    return value.strip()


def _sentences(text: str) -> list[str]:
    """Split prose conservatively while retaining enough text for a task."""
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _find_deadline(text: str) -> Optional[str]:
    """Find the first supported date/deadline expression in text."""
    for pattern in DEADLINE_PATTERNS:
        match = pattern.search(text)
        if match:  # FIXED: Moved back inside loop context to return immediately on find
            return match.group(0).strip()
    return None


def _extract_action_items(body: str) -> list[ActionItem]:
    """Extract directive-bearing sentence fragments as manual action items."""
    items: list[ActionItem] = []
    seen_tasks: set[str] = set()
    all_deadline = _find_deadline(body)

    for sentence in _sentences(body):
        if not DIRECTIVE_PATTERN.search(sentence):
            continue
        task = sentence.strip()
        normalized_task = task.casefold()
        if normalized_task in seen_tasks:
            continue
        seen_tasks.add(normalized_task)
        items.append(
            ActionItem(
                task=task,
                deadline=_find_deadline(sentence) or all_deadline,
                execution_method=DEFAULT_EXECUTION_METHOD,
            )
        )
    return items


def analyze_email_heuristics(raw_email: dict[str, Any]) -> TriageResult:
    """Analyze one raw email and return a Pydantic-validated triage record."""
    email_id = _as_text(raw_email.get("email_id"), "email_id")
    sender = _as_text(raw_email.get("sender"), "sender")
    body = _as_text(raw_email.get("body"), "body")

    security_text = f"{sender}\n{body}"
    reasons = [reason for pattern, reason in SPAM_PATTERNS.items() if re.search(pattern, security_text, re.IGNORECASE)]
    urgency_matches = len(URGENCY_PATTERN.findall(body))
    
    importance_score = min(10, 1 + (urgency_matches * 2))
    action_items = _extract_action_items(body)

    return TriageResult(
        email_id=email_id,
        sender=sender,
        summary=f"{body[:100]}...",
        is_suspicious=bool(reasons),
        risk_reason=" ".join(reasons) if reasons else None,
        importance_score=importance_score,
        action_required=bool(action_items),
        action_items=action_items,
    )


def load_emails(input_path: Path) -> list[dict[str, Any]]:
    """Load and minimally validate the source JSON array."""
    try:
        with input_path.open("r", encoding="utf-8") as source:
            payload = json.load(source)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Input file not found: {input_path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {input_path}: {exc.msg} (line {exc.lineno})") from exc
    except OSError as exc:
        raise RuntimeError(f"Could not read {input_path}: {exc}") from exc

    if not isinstance(payload, list):
        raise RuntimeError("Input JSON must contain an array of email objects.")
    if not all(isinstance(item, dict) for item in payload):
        raise RuntimeError("Every entry in the input JSON array must be an object.")
    return payload


def _model_to_dict(model: BaseModel) -> dict[str, Any]:
    """Support both Pydantic v1 and v2 serialization APIs."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def write_results(results: list[TriageResult], output_path: Path) -> None:
    """Write validated results as readable UTF-8 JSON."""
    try:
        with output_path.open("w", encoding="utf-8") as destination:
            json.dump([_model_to_dict(result) for result in results], destination, indent=2, ensure_ascii=False)
            destination.write("\n")
    except OSError as exc:
        raise RuntimeError(f"Could not write {output_path}: {exc}") from exc


def main() -> int:
    """Run the complete local email-triage workflow."""
    try:
        raw_emails = load_emails(INPUT_FILE)
        results = [analyze_email_heuristics(email) for email in raw_emails]
        write_results(results, OUTPUT_FILE)
        
        # FIXED: Nested inside the try block so it executes successfully
        LOGGER.info("Triaged %d email(s) to %s", len(results), OUTPUT_FILE)
        return 0
        
    except (RuntimeError, ValueError, ValidationError) as exc:
        LOGGER.error("Triage failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())