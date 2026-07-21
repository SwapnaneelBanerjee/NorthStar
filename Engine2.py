from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, Field, ValidationError

# =====================================================================
# LOGGING SETUP
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

# Suppress verbose SDK and HTTP logs
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_ENV_FILE = SCRIPT_DIR / "Key.env"
DEFAULT_INPUT_FILE = SCRIPT_DIR / "mock_emails.json"
DEFAULT_OUTPUT_FILE = SCRIPT_DIR / "triage_output.json"
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
DEFAULT_EXECUTION_METHOD = "Review email text to execute manual task."


# =====================================================================
# CONFIGURATION & ENVIRONMENT HELPERS
# =====================================================================
@dataclass
class RunConfig:
    input_file: Path = DEFAULT_INPUT_FILE
    output_file: Path = DEFAULT_OUTPUT_FILE
    env_file: Path = DEFAULT_ENV_FILE
    model: str = DEFAULT_MODEL
    max_retries: int = 3
    inter_call_delay: float = 6.0


def load_environment(env_file: Path) -> None:
    """Load environment variables from Key.env if available."""
    if env_file.exists():
        load_dotenv(dotenv_path=env_file)
    else:
        LOGGER.warning(
            "Could not find configuration file: %s. Falling back to system environment variables.",
            env_file,
        )


def require_api_key() -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "CRITICAL: GEMINI_API_KEY environment variable is missing. "
            "Please check your Key.env file or export it in your shell."
        )
    return api_key


def build_client() -> genai.Client:
    """Initialize the Google GenAI SDK Client."""
    require_api_key()
    try:
        return genai.Client()
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize Gemini client: {exc}") from exc


# =====================================================================
# DATA STRUCTURES & CONTRACTS
# =====================================================================
class ActionItem(BaseModel):
    """A concrete action identified in an email."""
    task: str = Field(description="The clear task description.")
    deadline: Optional[str] = Field(None, description="The deadline string or ISO format.")
    execution_method: str = Field(default=DEFAULT_EXECUTION_METHOD)
    safety_warning: Optional[str] = Field(
        None,
        description="Cautionary note if the task involves links, credentials, or urgent transfers from a flagged mail.",
    )


class LLMDeadlineExtraction(BaseModel):
    """Structured extraction schema for Gemini response enforcement."""
    summary: str = Field(description="One-sentence abstract of the email's core message.")
    is_suspicious: bool = Field(description="True if the email exhibits phishing, scam, or security anomalies.")
    risk_level: str = Field(description="Categorize risk level: 'NONE', 'LOW' (informal/hurried internal), 'HIGH' (phishing/threat).")
    risk_reason: Optional[str] = Field(None, description="Explanation of why this was flagged, or null if clean.")
    has_action_item: bool = Field(description="True if any task, request, or deadline is present.")
    primary_task: Optional[str] = Field(None, description="The primary action requested.")
    deadline_iso: Optional[str] = Field(None, description="Converted absolute deadline in YYYY-MM-DD HH:MM format.")
    deadline_raw_phrase: Optional[str] = Field(None, description="Literal words used for the deadline.")


class TriageResult(BaseModel):
    """Strict, serializable contract for output writing."""
    email_id: str = Field(min_length=1)
    sender: str = Field(min_length=1)
    summary: str
    is_suspicious: bool
    risk_level: str
    risk_reason: Optional[str] = None
    priority_value: Optional[float] = Field(
        default=None,
        description=(
            "Signed minutes remaining until the deadline (negative if overdue). "
            "None when the email has no actionable deadline -- this is intentionally "
            "distinct from 0.0, which would mean 'due this exact instant'."
        ),
    )
    deadline: Optional[str] = None
    action_required: bool
    action_items: list[ActionItem]


# =====================================================================
# DATA UTILITIES & ALGORITHMIC SCORE LOGIC
# =====================================================================
def _as_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Email field '{field_name}' must be a non-empty string.")
    return value.strip()


def _as_optional_text(value: Any) -> str:
    """Safely coerce a missing or None field to a stripped string."""
    return str(value or "").strip()


def _calculate_minutes_until_deadline(deadline_iso_str: Optional[str]) -> Optional[float]:
    """Calculate the exact number of minutes remaining until deadline (negative if overdue).

    Returns None -- not 0.0 -- when there is no deadline or it can't be parsed. Using 0.0
    as that fallback would be indistinguishable from a real deadline that is due this exact
    instant, silently making "no deadline" emails look exactly as urgent as "due right now"
    ones. None makes "no time pressure" explicit and keeps it out of the numeric ordering.
    """
    if not deadline_iso_str:
        return None

    deadline_datetime = None
    for parser in (
        lambda s: datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc),
        lambda s: datetime.fromisoformat(s.replace("Z", "")).replace(tzinfo=timezone.utc),
    ):
        try:
            deadline_datetime = parser(deadline_iso_str)
            break
        except (ValueError, TypeError):
            continue

    if deadline_datetime is None:
        return None

    now = datetime.now(timezone.utc)
    minutes_remaining = (deadline_datetime - now).total_seconds() / 60.0
    return round(minutes_remaining, 1)


# =====================================================================
# LLM QUERY LAYER
# =====================================================================
def query_llm_for_triage(
    client: genai.Client,
    model: str,
    sender: str,
    subject: str,
    timestamp: str,
    body: str,
    max_retries: int = 3,
    inter_call_delay: float = 6.0,
) -> LLMDeadlineExtraction:
    """Analyze email metrics using Gemini's structured output schema with rate limit protection."""
    current_time_context = datetime.now(timezone.utc).strftime("%A, %B %d, %Y at %I:%M %p UTC")

    system_instruction = (
        f"You are the core evaluation engine for Project NorthStar.\n"
        f"Current execution clock: {current_time_context}.\n\n"
        f"EVALUATION RULES:\n"
        f"1. RISK ASSESSMENT: Distinguish malicious external phishing (HIGH risk) from informal or hurried internal emails (LOW risk).\n"
        f"2. TASK EXTRACTION: ALWAYS extract tasks and deadlines even if the email is suspicious.\n"
        f"3. DATE PARSING: Use the email's sent timestamp ({timestamp}) as the anchor to convert relative dates "
        f"(e.g., 'tomorrow at 5 PM') into YYYY-MM-DD HH:MM."
    )

    user_content = f"Sent Timestamp: {timestamp}\nFrom: {sender}\nSubject: {subject}\nBody:\n{body}"

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=user_content,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=LLMDeadlineExtraction,
                    temperature=0.0,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
            
            # Successful call pacing to guarantee < 10 RPM on Free Tier. This runs
            # unconditionally, even after the very last email in a run (a few seconds
            # of harmless trailing delay), because we can't tell in advance whether
            # this attempt will actually be the final call -- if model_validate_json
            # below raises, we still need this pacing before the next retry.
            time.sleep(inter_call_delay)
            return LLMDeadlineExtraction.model_validate_json(response.text)

        except APIError as exc:
            error_code = getattr(exc, "code", None)
            is_rate_limit = error_code == 429 or "RESOURCE_EXHAUSTED" in str(exc)
            if is_rate_limit and attempt < max_retries:
                backoff_time = 15 * attempt
                LOGGER.warning(
                    "Free tier rate limit hit (429/RESOURCE_EXHAUSTED). Retrying attempt %d/%d in %ds...",
                    attempt, max_retries, backoff_time,
                )
                time.sleep(backoff_time)
            else:
                LOGGER.error("Gemini API failure on attempt %d/%d: %s", attempt, max_retries, exc)
                break

        except (ValidationError, ValueError) as exc:
            LOGGER.error("Response validation failed on attempt %d/%d: %s", attempt, max_retries, exc)
            if attempt >= max_retries:
                break

        except Exception as exc:
            LOGGER.error("Unexpected error during LLM query on attempt %d/%d: %s", attempt, max_retries, exc)
            break

    # Fallback default object if retries are exhausted or a non-retryable error occurs
    return LLMDeadlineExtraction(
        summary="Failed to parse summary due to API error.",
        is_suspicious=False,
        risk_level="NONE",
        has_action_item=False,
    )


# =====================================================================
# CORE PIPELINE LOGIC
# =====================================================================
def analyze_email_heuristics(
    raw_email: dict[str, Any],
    client: genai.Client,
    model: str,
    max_retries: int,
    inter_call_delay: float,
) -> TriageResult:
    email_id = _as_text(raw_email.get("email_id"), "email_id")
    sender = _as_text(raw_email.get("sender"), "sender")
    body = _as_text(raw_email.get("body"), "body")
    subject = _as_optional_text(raw_email.get("subject"))
    timestamp = _as_optional_text(raw_email.get("timestamp"))

    LOGGER.info("Processing email ID: %s", email_id)

    llm_data = query_llm_for_triage(
        client, model, sender, subject, timestamp, body,
        max_retries=max_retries, inter_call_delay=inter_call_delay,
    )

    action_items: list[ActionItem] = []
    priority_value: Optional[float] = None
    final_deadline_display = None

    if llm_data.has_action_item and llm_data.primary_task:
        final_deadline_display = llm_data.deadline_iso or llm_data.deadline_raw_phrase

        warning = None
        if llm_data.is_suspicious:
            warning = f"⚠️ {llm_data.risk_level} RISK: Verify sender via internal chat before opening links or taking action."

        action_items.append(
            ActionItem(
                task=llm_data.primary_task,
                deadline=llm_data.deadline_raw_phrase,
                execution_method=DEFAULT_EXECUTION_METHOD,
                safety_warning=warning,
            )
        )
        priority_value = _calculate_minutes_until_deadline(llm_data.deadline_iso)

    return TriageResult(
        email_id=email_id,
        sender=sender,
        summary=llm_data.summary,
        is_suspicious=llm_data.is_suspicious,
        risk_level=llm_data.risk_level,
        risk_reason=llm_data.risk_reason,
        priority_value=priority_value,
        deadline=final_deadline_display,
        action_required=bool(action_items),
        action_items=action_items,
    )


def load_emails(input_path: Path) -> list[dict[str, Any]]:
    try:
        with input_path.open("r", encoding="utf-8") as source:
            payload = json.load(source)
    except Exception as exc:
        raise RuntimeError(f"Failed to read input JSON file at {input_path}: {exc}") from exc

    if not isinstance(payload, list):
        raise RuntimeError("Input JSON must contain an array of objects.")
    return payload


def write_results(results: list[TriageResult], output_path: Path) -> None:
    try:
        with output_path.open("w", encoding="utf-8") as destination:
            dump_data = [r.model_dump() for r in results]
            json.dump(dump_data, destination, indent=2, ensure_ascii=False)
            destination.write("\n")
    except OSError as exc:
        raise RuntimeError(f"Could not write to output path {output_path}: {exc}") from exc


def run_pipeline(config: RunConfig) -> list[TriageResult]:
    load_environment(config.env_file)
    client = build_client()

    raw_emails = load_emails(config.input_file)
    results: list[TriageResult] = []
    failed_count = 0

    for index, raw_email in enumerate(raw_emails, start=1):
        try:
            result = analyze_email_heuristics(
                raw_email, client, config.model, config.max_retries, config.inter_call_delay,
            )
            results.append(result)
        except (ValueError, ValidationError) as exc:
            failed_count += 1
            email_id = raw_email.get("email_id", f"<unknown, index {index}>")
            LOGGER.error("Skipping email %s due to processing error: %s", email_id, exc)

    if failed_count:
        LOGGER.warning("%d of %d email(s) were skipped due to errors.", failed_count, len(raw_emails))

    write_results(results, config.output_file)
    return results


# =====================================================================
# CLI ENTRY POINT
# =====================================================================
def parse_args(argv: Optional[list[str]] = None) -> RunConfig:
    parser = argparse.ArgumentParser(description="Project NorthStar - Gemini email triage engine.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_FILE, help="Path to input emails JSON file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE, help="Path to write triage results JSON.")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE, help="Path to the .env file containing GEMINI_API_KEY.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Gemini model name to use.")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retry attempts per LLM call.")
    parser.add_argument("--delay", type=float, default=6.0, help="Seconds to wait between LLM calls (rate limiting).")
    args = parser.parse_args(argv)

    return RunConfig(
        input_file=args.input,
        output_file=args.output,
        env_file=args.env_file,
        model=args.model,
        max_retries=args.max_retries,
        inter_call_delay=args.delay,
    )


def main(argv: Optional[list[str]] = None) -> int:
    LOGGER.info("Starting Project NorthStar Engine [Gemini Triage Mode]...")
    config = parse_args(argv)
    try:
        results = run_pipeline(config)
        LOGGER.info("Triaged %d email(s) successfully to %s", len(results), config.output_file)
        return 0
    except (RuntimeError, ValueError, ValidationError) as exc:
        LOGGER.error("Execution processing halted: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
