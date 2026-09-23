"""Tests for skill loader and skill instructions injection."""

from __future__ import annotations

import os

from backend.app.agent.skills.loader import (
    _skill_instructions,
    extract_delivered_skills,
    get_skill_instructions,
    load_all_skills,
    load_skill_instructions,
    skill_delivery_marker,
    skill_guidance_block,
    strip_skill_guidance,
)
from backend.app.agent.tools.base import ToolResult
from backend.app.agent.tools.registry import create_list_capabilities_tool

# ---------------------------------------------------------------------------
# load_skill_instructions
# ---------------------------------------------------------------------------


def test_load_skill_instructions_reads_skill_md() -> None:
    """load_skill_instructions should read SKILL.md from the given directory."""
    skill_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "backend",
        "app",
        "integrations",
        "quickbooks",
    )
    content = load_skill_instructions(os.path.normpath(skill_dir))
    assert "QuickBooks" in content
    assert "qb_query" in content


def test_load_skill_instructions_missing_file(tmp_path: str) -> None:
    """load_skill_instructions should return empty string for missing SKILL.md."""
    content = load_skill_instructions(str(tmp_path))
    assert content == ""


# ---------------------------------------------------------------------------
# load_all_skills / get_skill_instructions
# ---------------------------------------------------------------------------


def test_load_all_skills_discovers_quickbooks() -> None:
    """load_all_skills should find the quickbooks skill package."""
    load_all_skills()
    assert "quickbooks" in _skill_instructions
    assert "qb_query" in _skill_instructions["quickbooks"]


def test_get_skill_instructions_returns_content() -> None:
    """get_skill_instructions should return SKILL.md content for known skills."""
    load_all_skills()
    content = get_skill_instructions("quickbooks")
    assert content is not None
    assert "QuickBooks" in content


def test_get_skill_instructions_returns_none_for_unknown() -> None:
    """get_skill_instructions should return None for unknown skill names."""
    assert get_skill_instructions("nonexistent_skill") is None


def test_load_all_skills_discovers_servicetitan() -> None:
    """load_all_skills should find the servicetitan integration SKILL.md."""
    load_all_skills()
    content = get_skill_instructions("servicetitan")
    assert content is not None
    assert content.strip() != ""
    # Pin a few load-bearing references so accidental deletes surface in CI.
    assert "ServiceTitan" in content
    assert "st_search_customers" in content
    assert "st_list_appointments" in content
    # Connecting moved to the web app (issue #1337); the SKILL must say so.
    assert "web app" in content


# ---------------------------------------------------------------------------
# list_capabilities integration
# ---------------------------------------------------------------------------


async def test_list_capabilities_includes_skill_instructions() -> None:
    """Looking up a category with a SKILL.md should include the SKILL guidance."""
    load_all_skills()
    tool = create_list_capabilities_tool({"quickbooks": "QB tools"})
    result: ToolResult = await tool.function(category="quickbooks")
    assert result.is_error is False
    assert "already loaded" in result.content.lower()
    assert "QuickBooks" in result.content
    assert "qb_query" in result.content
    assert "Common Workflows" in result.content


async def test_list_capabilities_without_skill_instructions() -> None:
    """Looking up a category without a SKILL.md should just show the guidance message."""
    tool = create_list_capabilities_tool({"other_category": "Some tools"})
    result: ToolResult = await tool.function(category="other_category")
    assert result.is_error is False
    assert "already loaded" in result.content.lower()
    # Should not contain skill instructions since "other_category" has no SKILL.md
    assert "SKILL" not in result.content


async def test_list_capabilities_listing_unchanged() -> None:
    """Listing categories (no category arg) should work as before."""
    tool = create_list_capabilities_tool({"quickbooks": "QB tools", "files": "File tools"})
    result: ToolResult = await tool.function(category=None)
    assert result.is_error is False
    assert "quickbooks" in result.content
    assert "files" in result.content


async def test_list_capabilities_unknown_category() -> None:
    """Unknown categories should still return an error."""
    tool = create_list_capabilities_tool({"quickbooks": "QB tools"})
    result: ToolResult = await tool.function(category="nonexistent")
    assert result.is_error is True
    assert "Unknown category" in result.content


async def test_list_capabilities_result_carries_delivery_marker() -> None:
    """SKILL.md delivery must be tagged so the agent loop can detect it in history."""
    load_all_skills()
    tool = create_list_capabilities_tool({"quickbooks": "QB tools"})
    result: ToolResult = await tool.function(category="quickbooks")
    assert skill_delivery_marker("quickbooks") in result.content
    assert extract_delivered_skills(result.content) == {"quickbooks"}


# ---------------------------------------------------------------------------
# delivery markers
# ---------------------------------------------------------------------------


def test_extract_delivered_skills_round_trip() -> None:
    """Markers embedded in tool-result text are recovered by extraction."""
    text = (
        f"did the thing\n\n{skill_delivery_marker('companycam')}\nguidance here\n"
        f"more output\n{skill_delivery_marker('calendar')}\nother guidance"
    )
    assert extract_delivered_skills(text) == {"companycam", "calendar"}


def test_extract_delivered_skills_ignores_plain_text() -> None:
    """Text without markers yields an empty set."""
    assert extract_delivered_skills("no markers here, just [brackets] and words") == set()


# ---------------------------------------------------------------------------
# stripping a delivered block
# ---------------------------------------------------------------------------

_GUIDANCE = "## QuickBooks\nAlways look up the customer first.\n\n- [x] a checklist line"


def test_guidance_block_is_delimited_on_both_sides() -> None:
    block = skill_guidance_block("quickbooks", _GUIDANCE)
    assert block == (
        f"\n\n[skill-guidance: quickbooks]\n{_GUIDANCE}\n[/skill-guidance: quickbooks]"
    )
    # Only the opening line counts as a delivery.
    assert extract_delivered_skills(block) == {"quickbooks"}


def test_strip_removes_a_delimited_block_and_keeps_the_result() -> None:
    result = "ok | Id: 643"
    assert strip_skill_guidance(result + skill_guidance_block("quickbooks", _GUIDANCE)) == result


def test_strip_keeps_text_after_a_delimited_block() -> None:
    """A delimited block is cut out wherever it sits, not to the end."""
    text = f"ok | Id: 643{skill_guidance_block('quickbooks', _GUIDANCE)}\ntrailing note"
    assert strip_skill_guidance(text) == "ok | Id: 643\ntrailing note"


def test_strip_removes_a_legacy_block_to_the_end() -> None:
    """Rows stored before the closing marker end with the guidance."""
    legacy = f"ok | Id: 643\n\n{skill_delivery_marker('quickbooks')}\n{_GUIDANCE}"
    assert strip_skill_guidance(legacy) == "ok | Id: 643"


def test_strip_leaves_results_without_guidance_alone() -> None:
    for text in (
        "ok | Id: 643",
        "",
        "mentions [skill-guidance: quickbooks] inline, not on its own line",
    ):
        assert strip_skill_guidance(text) == text
    assert extract_delivered_skills(strip_skill_guidance(_GUIDANCE)) == set()


async def test_list_capabilities_result_strips_to_its_own_text() -> None:
    load_all_skills()
    tool = create_list_capabilities_tool({"quickbooks": "QB tools"})
    result: ToolResult = await tool.function(category="quickbooks")
    stripped = strip_skill_guidance(result.content)
    assert stripped.startswith('Tools for "quickbooks" are already loaded')
    assert extract_delivered_skills(stripped) == set()
    instructions = get_skill_instructions("quickbooks")
    assert instructions is not None and instructions not in stripped
