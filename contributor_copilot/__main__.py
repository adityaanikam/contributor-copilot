"""Ask one question from the command line: python -m contributor_copilot "question" """

from __future__ import annotations

import argparse

from contributor_copilot.config import Settings
from contributor_copilot.service import AskResult, build_copilot


def render(result: AskResult) -> str:
    lines = [result.answer, ""]
    for citation in result.citations:
        lines.append(f"[{citation.number}] {citation.location}  ({citation.section})")
        lines.append(f"    {citation.url}")
    if result.citations and not result.grounded:
        lines.append("")
        lines.append("Note: the grounding check did not confirm this answer. Check the sources.")
    return "\n".join(lines).rstrip()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ask Contributor Copilot a question.")
    parser.add_argument("question")
    args = parser.parse_args(argv)
    copilot = build_copilot(Settings.from_env())
    print(render(copilot.ask(args.question)))


if __name__ == "__main__":
    main()
