"""Skill loader: reads SKILL.md files from skill packages.

Skills are documentation-only packages that provide LLM-facing instructions
for a group of related tools. They are injected into the conversation context
when a specialist tool category is activated, following the OpenClaw pattern
of separating documentation (skills) from execution (tools).
"""

from __future__ import annotations

import importlib
import logging
import os
import pkgutil
import re

logger = logging.getLogger(__name__)

# Marker line prepended to SKILL.md content whenever it is delivered into a
# tool result (via list_capabilities or first-use auto-injection). Scanning
# reloaded history for these markers tells the agent which categories'
# guidance is already in context, so trimming or stripping a delivery re-arms
# injection. The closing marker (``[/skill-guidance: <name>]``) does not match
# this pattern, so only the opening line counts as a delivery.
_SKILL_MARKER_RE = re.compile(r"\[skill-guidance: ([A-Za-z0-9_-]+)\]")

# A delivered block, as :func:`skill_guidance_block` writes it: a blank line,
# the opening marker on its own line, the SKILL.md, and the closing marker.
_DELIMITED_BLOCK_RE = re.compile(
    r"\n\n\[skill-guidance: ([A-Za-z0-9_-]+)\]\n.*?\n\[/skill-guidance: \1\]", re.DOTALL
)
# Rows stored before the closing marker existed carry only the opening line.
# Their block was always appended last, so it runs to the end of the result.
_LEGACY_BLOCK_START_RE = re.compile(r"\n\n\[skill-guidance: [A-Za-z0-9_-]+\]\n")

# Mapping of factory name -> SKILL.md content, populated by load_all_skills().
_skill_instructions: dict[str, str] = {}


def load_skill_instructions(skill_dir: str) -> str:
    """Read SKILL.md from a skill's package directory.

    Returns empty string if the file is not found.
    """
    path = os.path.join(skill_dir, "SKILL.md")
    try:
        with open(path) as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def load_all_skills() -> None:
    """Discover all skill packages and load their SKILL.md content.

    Scans two locations:

    1. ``backend.app.agent.skills.*`` -- skill-only packages
    2. ``backend.app.integrations.*`` -- self-contained integration packages

    Each sub-package that contains a SKILL.md file is loaded. The package
    name (e.g. ``quickbooks``) is used as the key, matching the tool
    factory registration name.
    """
    _scan_package("backend.app.agent.skills")
    _scan_package("backend.app.integrations")


def _scan_package(package_path: str) -> None:
    """Scan a top-level package for sub-packages containing SKILL.md."""
    try:
        package = importlib.import_module(package_path)
    except ModuleNotFoundError:
        logger.debug("Package %s not found, skipping skill scan", package_path)
        return

    for _, name, is_pkg in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
        if not is_pkg:
            continue
        mod = importlib.import_module(name)
        mod_dir = os.path.dirname(mod.__file__ or "")
        content = load_skill_instructions(mod_dir)
        if content:
            # Use the short package name (e.g. "quickbooks") as the key
            short_name = name.rsplit(".", 1)[-1]
            _skill_instructions[short_name] = content
            logger.debug("Loaded skill instructions for %r (%d chars)", short_name, len(content))


def get_skill_instructions(factory_name: str) -> str | None:
    """Return the SKILL.md content for a factory name, or None if not found."""
    return _skill_instructions.get(factory_name)


def skill_delivery_marker(factory_name: str) -> str:
    """Return the marker line that tags delivered skill guidance in a tool result."""
    return f"[skill-guidance: {factory_name}]"


def skill_guidance_block(factory_name: str, instructions: str) -> str:
    """Return the text appended to a tool result to deliver a category's SKILL.md.

    The block is delimited on both sides so :func:`strip_skill_guidance` can
    remove it without touching the result it rides on.
    """
    return (
        f"\n\n{skill_delivery_marker(factory_name)}\n{instructions}"
        f"\n[/skill-guidance: {factory_name}]"
    )


def strip_skill_guidance(text: str) -> str:
    """Return *text* with any delivered SKILL.md block removed.

    Blocks written by :func:`skill_guidance_block` are removed wherever they
    sit. A result with none of those may be a row stored before the closing
    marker existed, whose block has only the opening line and is always the
    tail of the result: it is cut from the last opening line to the end.
    """
    stripped, count = _DELIMITED_BLOCK_RE.subn("", text)
    if count:
        return stripped
    starts = [m.start() for m in _LEGACY_BLOCK_START_RE.finditer(text)]
    return text[: starts[-1]] if starts else text


def extract_delivered_skills(text: str) -> set[str]:
    """Return factory names whose delivery marker appears in *text*."""
    return set(_SKILL_MARKER_RE.findall(text))
