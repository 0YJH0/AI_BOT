"""Read-only checks of report links, documented CLI flags and evidence hashes.

This intentionally does not launch Isaac, train a policy or claim fresh-machine
demo verification. It uses only the Python standard library.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "docs/reports_20261013"


def main():
    documents = [ROOT / "README.md", *sorted(REPORTS.glob("*.md"))]
    links = commands = 0
    launcher_flags = {"--headless", "--device", "--rendering_mode", "--enable_cameras"}
    for document in documents:
        content = document.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", content):
            if target.startswith(("http:", "https:", "#")):
                continue
            target = target.split("#")[0]
            assert (document.parent / target).exists(), (document, target)
            links += 1
        for command in re.findall(r"^python (scripts[\\/][^\s]+\.py)([^\n]*)", content, re.M):
            relative, tail = command
            script = ROOT / relative.replace("\\", "/")
            assert script.is_file(), script
            sources = [script]
            if script.name == "serve_smolvla_windows_v044_v4.py":
                sources.append(ROOT / "scripts/serve_smolvla.py")
            if script.name == "run_vla_parallel_probe.py":
                sources.append(ROOT / "scripts/run_vla_benchmark.py")
            flags = set(launcher_flags)
            for source in sources:
                tree = ast.parse(source.read_text(encoding="utf-8-sig"))
                flags.update(n.value for n in ast.walk(tree)
                             if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith("--"))
            passed = set(re.findall(r"(?<!\w)--[\w-]+", tail))
            assert passed <= flags, (document.name, script.name, sorted(passed - flags))
            commands += 1
    snapshot = json.loads((REPORTS / "evidence_snapshot.json").read_text(encoding="utf-8"))
    for source in snapshot["sources"]:
        original = ROOT / source["source"]
        if original.is_file():
            assert hashlib.sha256(original.read_bytes()).hexdigest() == source["source_sha256"], original
    benchmark = next(s["snapshot"] for s in snapshot["sources"] if s["source"].endswith("comparison_summary.json"))
    assert benchmark["status"] == "COMPLETE"
    assert [r["successes"] for r in benchmark["results"]] == [10, 8, 14, 7]
    assert all(r["episodes"] == 64 for r in benchmark["results"])
    assert len({r["scene_manifest_sha256"] for r in benchmark["results"]}) == 1
    for stem in ("01_project_challenges", "02_project_issue_report", "03_project_overview", "04_project_one_hour_talk"):
        assert (ROOT / "output/pdf" / f"{stem}.pdf").stat().st_size > 10000
    print(json.dumps({"status": "PASS", "documents": len(documents), "local_links": links,
                      "documented_commands_static_checked": commands,
                      "evidence_sources": len(snapshot["sources"]),
                      "note": "Static CLI check only; no new simulation or training."}, ensure_ascii=False))


if __name__ == "__main__":
    main()
