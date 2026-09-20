"""The defense must never learn who is who. Judges check this first."""

import pathlib
import re
import subprocess
import sys

DEFENSE = pathlib.Path(__file__).resolve().parents[1] / "antidote" / "defense"

# Words that would mean the defense is reading the answer key rather than the
# updates. "inspect" contains "spec", so match whole words only.
FORBIDDEN = [
    "spec",
    "specs",
    "role",
    "roles",
    "attacker",
    "attackers",
    "backdoor",
    "poison",
    "poisoned",
    "malicious",
    "compromised",
]


def defense_sources():
    files = sorted(DEFENSE.glob("*.py"))
    assert files, "no defense source found"
    return files


def test_the_defense_source_never_mentions_who_is_bad():
    for path in defense_sources():
        source = path.read_text(encoding="utf-8").lower()
        for word in FORBIDDEN:
            assert not re.search(rf"\b{word}\b", source), f"{path.name} says {word}"


def test_the_defense_never_imports_the_attack_code():
    for path in defense_sources():
        source = path.read_text(encoding="utf-8")
        assert "antidote.attacks" not in source
        assert not re.search(r"^\s*(from|import)\s+\S*attacks", source, re.M)


def test_importing_the_defense_does_not_load_the_attack_code():
    code = (
        "import sys; import antidote.defense; "
        "loaded = [m for m in sys.modules if 'attacks' in m]; "
        "assert not loaded, loaded"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=DEFENSE.parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
