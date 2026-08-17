"""
sequence_pipeline.py
=====================
Adds a third capability to H3 Prompt Studio, on top of the existing per-mode
prompt builders: turn a set of reference images into a short story, then
split that story into N successive video sequences, each ready to be
expanded into a standalone H3 Ref2VA prompt.

Two LLM passes, both using the app's existing LLMClient (same
model/system_prompt/user_prompt/on_token interface as the rest of the app —
see llm_client.py):

  Pass A - STORY: the reference library (image roles + descriptions, as
  already captured by Ref2VA's "Add a reference" dialog / vision-model
  description) plus a short premise are expanded into a coherent short
  narrative. This reuses the "story forge" pattern (idea notes -> LLM
  expansion) but folds it into a single pass, since the target output here
  is a sequencing aid, not literary prose.

  Pass B - SEQUENCE BREAKDOWN: the story text + reference library + camera
  vocabulary are given to the LLM, which returns a JSON array of exactly N
  sequence objects. For each sequence the LLM itself decides: which
  reference labels to use, whether there is dialogue (and its text/speaker/
  language), and the camera movement — exactly the three requirements asked
  for. Extra free-text instructions from the user are appended to the
  breakdown prompt and are also carried into every generated prompt.

The JSON-repair helpers are ported from story_forge/story_pipeline.py
(same truncation failure mode: local models cut off by max_tokens).
"""

from __future__ import annotations

import json
import re
from typing import Callable, Optional

from llm_client import LLMClient, LLMError


# --------------------------------------------------------------------------- #
# Pass A - story generation from the reference library
# --------------------------------------------------------------------------- #

STORY_FROM_REFS_SYSTEM_PROMPT = """\
You are a screenwriter. You are given a library of reference assets (images,
and optionally short video/audio clips) already annotated with a role and a
visual description, plus a short premise from the author. Write a short,
coherent narrative (roughly 200-500 words) that:
  - uses every reference asset that has an obvious dramatic role (do not
    ignore assets just because the premise is thin — invent connective
    tissue between them),
  - stays visual and concrete (actions, settings, objects, gestures) since
    this text will later be cut into short video sequences,
  - includes at least a little spoken dialogue where it feels natural, kept
    short and quotable,
  - has a clear beginning, middle and end (even a very small one).
Refer to reference assets naturally by what they depict, not by their file
labels. Write only the narrative itself - no title, no meta-commentary, no
markdown."""


def _reference_library_text(references: list[dict]) -> str:
    lines = []
    counters = {"Picture": 0, "Video": 0, "Audio": 0}
    if not references:
        return "(no reference asset provided)"
    for ref in references:
        counters[ref["type"]] += 1
        tag = f"<{ref['type']} {counters[ref['type']]}>"
        role = ref.get("role") or "(no role specified)"
        desc = ref.get("description") or "(no description)"
        lines.append(f"- {tag}: role = {role} | description = {desc}")
    return "\n".join(lines)


def build_story_user_prompt(references: list[dict], premise: str, language: str) -> str:
    return (
        f"Reference library:\n{_reference_library_text(references)}\n\n"
        f"Author's premise (may be brief or empty - invent freely if so): "
        f"{premise.strip() or '(none given - invent a premise consistent with the references)'}\n\n"
        f"Write the narrative in {language or 'English'}."
    )


def generate_story(
    client: LLMClient,
    model: str,
    references: list[dict],
    premise: str,
    language: str,
    temperature: float = 0.85,
    on_token: Optional[Callable[[str], None]] = None,
) -> str:
    """Pass A. Returns the full story text (also streamed via on_token if given)."""
    user_prompt = build_story_user_prompt(references, premise, language)
    return client.chat(
        model=model,
        system_prompt=STORY_FROM_REFS_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=temperature,
        on_token=on_token,
    )


# --------------------------------------------------------------------------- #
# Pass B - sequence breakdown
# --------------------------------------------------------------------------- #

SEQUENCE_BREAKDOWN_SYSTEM_PROMPT = """\
You are a video director splitting a short story into a fixed number of
successive video sequences (each sequence will later become one standalone
H3 Ref2VA generation, a few seconds long, so keep each sequence's scope to
one continuous beat of action).

You are given: the story text, the full reference asset library (with the
exact labels to use - <Picture N>, <Video N>, <Audio N>), the camera-motion
vocabulary allowed, the requested number of sequences, and optional extra
instructions from the author that apply to every sequence.

For EACH sequence, decide yourself:
  - which reference label(s) from the library are actually relevant to it
    (use only labels that exist in the library; it is fine for a sequence to
    use zero, one, or several). CRITICAL: list EVERY reference whose subject
    (character, creature, object, location, style...) is depicted or implied
    in that sequence's own brief text - not just the most prominent one. If
    your brief mentions a character AND a place, and both have a matching
    reference in the library, BOTH labels must appear in "references". A
    sequence's reference list must never omit a reference whose subject you
    just wrote into the brief.
  - whether it has dialogue: if yes, give a short speaker description, the
    dialogue language, and the exact line(s) (keep them brief - one or two
    short sentences per sequence, never invent long speeches),
  - the single best camera movement for it, chosen ONLY from the provided
    camera-motion vocabulary, plus a short amplitude/speed qualifier (e.g.
    "low amplitude, slow speed").

Cover the whole story in order, one continuous beat per sequence, without
skipping or repeating story beats. Respect the requested sequence count
exactly.

Before outputting, re-read each sequence's brief against the reference
library one more time and add any missing label whose subject appears in
that brief.

Output ONLY a single JSON array (no markdown fences, no commentary) with
exactly N objects, each shaped like this:
{
  "index": 1,
  "title": "3-6 word label for this sequence",
  "brief": "1-3 sentences describing what happens in this sequence, concrete and visual",
  "references": ["<Picture 1>", "<Picture 2>"],
  "dialogue": {
    "present": true,
    "speaker": "short speaker description, or empty string if present=false",
    "language": "language name, or empty string if present=false",
    "text": "exact line(s), or empty string if present=false",
    "voiceover": false
  },
  "camera_movement": "e.g. Pan Left, low amplitude, slow speed",
  "onscreen_text": ""
}"""


def _reference_library_text_for_breakdown(references: list[dict]) -> str:
    return _reference_library_text(references)


def build_breakdown_user_prompt(
    story_text: str,
    references: list[dict],
    n_sequences: int,
    camera_motions: list[str],
    extra_instructions: str,
    duration_per_sequence: int,
) -> str:
    lines = [
        f"STORY:\n{story_text.strip()}\n",
        f"REFERENCE LIBRARY:\n{_reference_library_text_for_breakdown(references)}\n",
        f"ALLOWED CAMERA MOTIONS: {', '.join(camera_motions)}\n",
        f"NUMBER OF SEQUENCES REQUESTED: {n_sequences}",
        f"APPROXIMATE DURATION PER SEQUENCE: {duration_per_sequence} seconds",
    ]
    if extra_instructions and extra_instructions.strip():
        lines.append(f"EXTRA INSTRUCTIONS (apply to every sequence): {extra_instructions.strip()}")
    lines.append("\nReturn the JSON array now.")
    return "\n".join(lines)


SEQUENCE_MAX_TOKENS = 4096


def generate_sequence_breakdown(
    client: LLMClient,
    model: str,
    story_text: str,
    references: list[dict],
    n_sequences: int,
    camera_motions: list[str],
    extra_instructions: str = "",
    duration_per_sequence: int = 8,
    temperature: float = 0.6,
) -> list[dict]:
    """Pass B. Returns a list of exactly n_sequences sequence dicts.
    Raises LLMError / ValueError on failure after one repair/retry attempt,
    mirroring story_forge's generate_blueprint() truncation handling."""
    user_prompt = build_breakdown_user_prompt(
        story_text, references, n_sequences, camera_motions, extra_instructions, duration_per_sequence
    )
    raw = client.chat(
        model=model,
        system_prompt=SEQUENCE_BREAKDOWN_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=temperature,
    )
    try:
        seqs = _parse_json_sequences(raw, n_sequences)
    except (json.JSONDecodeError, ValueError):
        try:
            repaired = _repair_truncated_json_array(raw)
            seqs = _parse_json_sequences(repaired, n_sequences)
        except (json.JSONDecodeError, ValueError):
            retry_prompt = (
                user_prompt
                + "\n\nYour previous answer was cut off or invalid JSON. Return the same "
                "breakdown again, with shorter text fields if needed, as a single complete "
                "and valid JSON array of exactly "
                f"{n_sequences} objects - nothing else."
            )
            raw_retry = client.chat(
                model=model,
                system_prompt=SEQUENCE_BREAKDOWN_SYSTEM_PROMPT,
                user_prompt=retry_prompt,
                temperature=max(0.3, temperature - 0.2),
            )
            try:
                seqs = _parse_json_sequences(raw_retry, n_sequences)
            except (json.JSONDecodeError, ValueError):
                repaired_retry = _repair_truncated_json_array(raw_retry)
                seqs = _parse_json_sequences(repaired_retry, n_sequences)  # let remaining errors surface

    return [_reconcile_sequence_references(seq, references) for seq in seqs]


REQUIRED_SEQUENCE_KEYS = ["index", "title", "brief", "references", "dialogue", "camera_movement"]


def _parse_json_sequences(raw: str, n_expected: int) -> list[dict]:
    text = _strip_to_json_array(raw)
    data = json.loads(text)
    if not isinstance(data, list) or not data:
        raise ValueError("La réponse du LLM n'est pas un tableau JSON non vide.")
    for i, seq in enumerate(data, start=1):
        missing = [k for k in REQUIRED_SEQUENCE_KEYS if k not in seq]
        if missing:
            raise ValueError(f"Séquence {i} incomplète, champs manquants : {missing}")
        seq.setdefault("onscreen_text", "")
        seq["dialogue"].setdefault("present", False)
        seq["dialogue"].setdefault("speaker", "")
        seq["dialogue"].setdefault("language", "")
        seq["dialogue"].setdefault("text", "")
        seq["dialogue"].setdefault("voiceover", False)
    return data


# Common short French/English function words, excluded from the keyword-overlap
# safety net below so they never count as a "shared subject" between a
# reference's description and a sequence's brief.
_STOPWORDS = {
    "dans", "avec", "vers", "sous", "pour", "cette", "cette", "cette", "leurs",
    "elle", "elles", "ils", "être", "etre", "avoir", "fait", "sont", "vers",
    "puis", "alors", "tandis", "tandis", "there", "their", "which", "while",
    "these", "those", "about", "would", "could", "should", "before", "after",
}


def _keywords(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-zàâäéèêëïîôöùûüç]{5,}", (text or "").lower())} - _STOPWORDS


def _reconcile_sequence_references(seq: dict, references: list[dict]) -> dict:
    """Safety net for the recurring failure mode where the LLM writes a
    subject (a character, a place...) into a sequence's brief but forgets to
    list the matching reference label in that sequence's "references" array.
    Re-adds any reference tag whose role/description shares a meaningful
    keyword with the brief text. This can only ADD tags, never remove ones
    the LLM explicitly chose."""
    brief_keywords = _keywords(f"{seq.get('title', '')} {seq.get('brief', '')}")
    if not brief_keywords:
        return seq

    existing = set(seq.get("references") or [])
    counters = {"Picture": 0, "Video": 0, "Audio": 0}
    for ref in references:
        counters[ref["type"]] += 1
        tag = f"<{ref['type']} {counters[ref['type']]}>"
        if tag in existing:
            continue
        ref_keywords = _keywords(f"{ref.get('role', '')} {ref.get('description', '')}")
        if ref_keywords & brief_keywords:
            existing.add(tag)

    def _sort_key(tag: str):
        kind, num = tag.strip("<>").split()
        return (kind, int(num))

    seq["references"] = sorted(existing, key=_sort_key)
    return seq


def _strip_to_json_array(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    text = text.strip()
    if not text.startswith("["):
        start = text.find("[")
        if start != -1:
            text = text[start:]
    return text


def _repair_truncated_json_array(raw: str) -> str:
    """Same boundary-scanning repair as story_forge's _repair_truncated_json,
    adapted for an array root instead of an object root."""
    text = _strip_to_json_array(raw)

    stack = []
    in_string = False
    escape = False
    last_safe_end = 0

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
            last_safe_end = i + 1
        elif ch == ",":
            last_safe_end = i

    if not in_string and not stack:
        return text

    repaired = text[:last_safe_end].rstrip()
    repaired = repaired.rstrip(",")
    closers = {"{": "}", "[": "]"}
    for opener in reversed(stack):
        repaired += closers[opener]
    return repaired


# --------------------------------------------------------------------------- #
# Turning a sequence dict into a Ref2VA storyboard brief
# --------------------------------------------------------------------------- #

def sequence_to_brief(
    seq: dict,
    n_total: int,
    reference_library_block: str,
    overall_story: str,
    duration_per_sequence: int,
    style: str,
    extra_instructions: str = "",
) -> str:
    """Builds one (label, brief) entry compatible with H3StudioApp.run_generation_sequence,
    in the same spirit as Ref2VATab._on_generate_storyboard()."""
    dlg = seq.get("dialogue", {})
    lines = [
        f"STORYBOARD SCENE {seq['index']}/{n_total} — treat as its own standalone H3 Ref2VA prompt "
        "(consistent with the same reference library across all scenes).",
        "ISOLATION RULE (critical): the prompt you write must depict ONLY what is described in "
        "\"BRIEF FOR THIS SCENE\" and \"DIALOGUE\" below. The OVERALL STORY line beneath is background "
        "context to keep character/setting consistency ONLY — never pull actions, dialogue, spoken "
        "lines, or events from it that are not explicitly part of this scene's own brief/dialogue. "
        "Do not anticipate later scenes or recap earlier ones.",
        f"OVERALL STORY (background context only — do not source scene content from this): "
        f"{overall_story.strip()[:2000]}",
        f"BRIEF FOR THIS SCENE ONLY (develop into the full prompt — nothing from other scenes): {seq['brief']}",
        f"VISUAL STYLE: {style}",
        f"TARGET DURATION: {duration_per_sequence} seconds",
        f"CAMERA MOVEMENT FOR THIS SCENE: {seq.get('camera_movement', '')}",
        "",
        reference_library_block,
        f"PRIMARY REFERENCES FOR THIS SCENE: "
        f"{', '.join(seq.get('references') or []) or '(none - use scene description only)'}",
        "",
    ]
    if dlg.get("present"):
        lines.append("DIALOGUE: yes")
        if dlg.get("speaker"):
            lines.append(f"  Speaker description: {dlg['speaker']}")
        lines.append(f"  Language: {dlg.get('language') or 'English'}")
        if dlg.get("text"):
            lines.append(f"  Exact line(s) to preserve verbatim: {dlg['text']}")
        if dlg.get("voiceover"):
            lines.append(
                "  This is an off-screen voiceover, not on-screen speech - the speaking "
                "character's lips stay closed."
            )
    else:
        lines.append(
            "DIALOGUE: none - do not invent spoken dialogue, and do not reuse dialogue lines "
            "belonging to other scenes of the story."
        )

    if seq.get("onscreen_text"):
        lines.append(f"ON-SCREEN TEXT: {seq['onscreen_text']}")

    if extra_instructions and extra_instructions.strip():
        lines.append(f"\nEXTRA INSTRUCTIONS: {extra_instructions.strip()}")

    return "\n".join(lines)
