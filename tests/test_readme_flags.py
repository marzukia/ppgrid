"""Docs smoke (issue #100 item 5, s23 P0: examples match actual behaviour).

Asserts the README "CLI Options" table rows match the flag set of the real
`ppgrid --help`. Both sides are derived at test time — the help text is
captured from the live parser (same capture pattern as
tests/test_cli_help.py) and the table is parsed from README.md. There is no
hand-maintained flag list in this file: a new parser flag, a renamed flag, or
a stale table row fails the comparison.

The argparse meta options (`-h`, `--help`, `--version`) are excluded from the
table contract: argparse injects them and the table documents the pipeline
flags plus the `input` positional.

Scope note (accepted, review of #121): the comparison is on the flag SET
(first table column) only. A row that documents the wrong description for a
correctly-named flag is out of scope — a description check would need either a
hand-maintained list of expected texts or fuzzy matching, both of which run
against the no-hand-maintained-list rule above.
"""

import io
import re
from pathlib import Path

from ppgrid.pipeline import _build_parser

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
TABLE_HEADING = "### CLI Options"

# argparse-injected, not pipeline options (see module docstring).
META_OPTIONS = {"-h", "--help", "--version"}

# An option description line in help output: exactly two leading spaces, then
# an option string. Usage continuation lines are indented to the column after
# "usage: ppgrid ", group titles sit at column 0, wrapped help text is
# indented to the help column, and the epilog's verbatim lines start with
# '#', 'ppgrid ', or a letter/digit — none of them match.
_OPTION_LINE = re.compile(r"^  -{1,2}[A-Za-z][A-Za-z0-9-]*")
_FLAG_TOKEN = re.compile(r"^-{1,2}[A-Za-z][A-Za-z0-9-]*$")


def _help_text() -> str:
    """Capture the real --help output from the live parser."""
    buf = io.StringIO()
    _build_parser().print_help(file=buf)
    return buf.getvalue()


def _option_line_tokens(line: str) -> list[str]:
    """Flag tokens of one help option line: the dash tokens before the help text.

    argparse renders the metavar per option string on Python 3.11/3.12
    (`-o OUT, --out OUT`) but once, shared, on 3.13+ (`-o, --out OUT`). So
    after a flag token skip AT MOST ONE following non-flag token (that flag's
    metavar) and keep scanning; break only on TWO consecutive non-flag tokens,
    which marks the start of the help text. That sees every alias on all four
    interpreters while keeping dash strings inside help text out.
    Alias lists (`-o, --out`) yield every alias.
    """
    out: list[str] = []
    seen_flag = False
    after_metavar = False
    for tok in line.split():
        clean = tok.rstrip(",")
        if _FLAG_TOKEN.match(clean):
            out.append(clean)
            seen_flag = True
            after_metavar = False
        elif seen_flag and not after_metavar:
            after_metavar = True  # this flag's metavar; keep scanning
        else:
            break  # non-flag before any flag, or two consecutive non-flags
    return out


def _flags_from_help_text(text: str) -> set[str]:
    """Every option string the real --help output renders (all aliases)."""
    flags: set[str] = set()
    for line in text.splitlines():
        if _OPTION_LINE.match(line):
            flags.update(_option_line_tokens(line))
    return flags


def _positional_from_help_text(text: str) -> str | None:
    """Name of the first positional in the 'positional arguments:' block."""
    m = re.search(r"^positional arguments:\n  (\S+)\s", text, re.MULTILINE)
    return m.group(1) if m else None


def _parser_flags() -> tuple[set[str], set[str]]:
    """(option strings, positional names) straight off the live parser."""
    options: set[str] = set()
    positionals: set[str] = set()
    for action in _build_parser()._actions:
        if action.option_strings:
            options.update(action.option_strings)
        else:
            positionals.add(action.dest)
    return options, positionals


def _readme_table_flags() -> set[str]:
    """Return the flags named in the first column of the README 'CLI Options' table."""
    text = README.read_text(encoding="utf-8")
    assert TABLE_HEADING in text, f"README.md: {TABLE_HEADING!r} section not found"
    section = text.split(TABLE_HEADING, 1)[1].split("\n### ", 1)[0]
    flags: set[str] = set()
    rows = 0
    for raw in section.splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            continue
        first = line.strip("|").split("|", 1)[0].strip()
        if first == "Flag" or not first or set(first) <= set("-: "):
            continue  # header row + separator row
        rows += 1
        flags.update(cell.strip().strip("`") for cell in first.split(","))
    assert rows >= 10, f"CLI Options table parsed only {rows} rows; table layout changed?"
    return flags


# ---------------------------------------------------------------------------
# Help-text parse is faithful to the parser (guards the extraction below)
# ---------------------------------------------------------------------------


def test_help_text_renders_every_parser_flag() -> None:
    """The --help text parse sees exactly what the live parser defines."""
    text = _help_text()
    rendered = _flags_from_help_text(text)
    defined_options, defined_positionals = _parser_flags()
    assert rendered == defined_options, (
        "help-text parse drifted from the parser: "
        f"only-in-text={sorted(rendered - defined_options)} "
        f"only-in-parser={sorted(defined_options - rendered)}"
    )
    assert _positional_from_help_text(text) in defined_positionals


# ---------------------------------------------------------------------------
# README 'CLI Options' table == real --help flag set
# ---------------------------------------------------------------------------


def test_readme_flag_table_matches_help() -> None:
    """README 'CLI Options' rows == the real --help flag set (issue #100 item 5).

    Every pipeline flag + the `input` positional the help renders must have a
    table row, and every table row must be a real flag: one-sided drift (a
    new flag, a rename, a stale row) fails this test.
    """
    text = _help_text()
    rendered = _flags_from_help_text(text)
    positional = _positional_from_help_text(text)
    expected = (rendered - META_OPTIONS) | ({positional} if positional else set())

    readme = _readme_table_flags()
    missing = sorted(expected - readme)
    stale = sorted(readme - expected)
    assert not missing, f"flags in --help with no README table row: {missing}"
    assert not stale, f"README table rows not in --help: {stale}"


# ---------------------------------------------------------------------------
# Support + Known limitations sections exist and say the right things
# ---------------------------------------------------------------------------


def test_readme_support_section() -> None:
    """The Support section: Py 3.11-3.14, all three OSes named, SUPPORT.md linked."""
    text = README.read_text(encoding="utf-8")
    assert "## Support" in text, "README.md: '## Support' section not found"
    section = text.split("## Support", 1)[1].split("\n## ", 1)[0]
    assert "3.11" in section
    assert "3.14" in section
    assert "SUPPORT.md" in section, "Support section must link SUPPORT.md (feedback path)"
    for os_name in ("Linux", "macOS", "Windows"):
        assert os_name in section, f"Support section does not name {os_name}"
    win_line = next((ln for ln in section.splitlines() if "Windows" in ln), "")
    assert "unsupported" in win_line.lower(), f"Windows must be claimed unsupported: {win_line!r}"


def test_readme_known_limitations() -> None:
    """Known limitations: both 0.4.2 items present, each with a one-line workaround."""
    text = README.read_text(encoding="utf-8")
    assert "## Known limitations" in text, "README.md: '## Known limitations' section not found"
    section = text.split("## Known limitations", 1)[1].split("\n## ", 1)[0]
    assert "not a spatially exact interpolator" in section, "IDW-approximation limitation missing"
    assert "visualisation, not downstream calculation" in section
    assert "shared-field budget, not process RSS" in section, "RAM-cap semantics limitation missing"
    assert "per-box" in section, "per-box is the only genuine low-RAM mode; name it"
    assert section.count("Workaround") >= 2, "each limitation needs a one-line workaround"
