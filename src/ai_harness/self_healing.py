from __future__ import annotations

import json
import re
from hashlib import sha256
from typing import List

from .models import CommandResult, RepairPlan, ReviewDecision


class SelfHealingPlanner:
    """Turn deterministic review findings into a bounded, repeat-aware repair plan."""

    def build(
        self,
        review: ReviewDecision,
        verification: List[CommandResult],
        history: List[RepairPlan],
    ) -> RepairPlan:
        fingerprint = self.failure_fingerprint(review, verification)
        repeat_count = sum(
            item.failure_fingerprint == fingerprint for item in history
        )
        failed_commands = [
            result.command
            for result in verification
            if result.exit_code != 0 or result.timed_out
        ]
        instructions = []
        for finding in review.findings:
            if finding.category == "timeout":
                instructions.append(
                    f"Investigate why `{finding.command}` timed out; make the smallest bounded fix."
                )
            elif finding.category == "verification":
                evidence = finding.evidence[-1200:]
                instructions.append(
                    f"Fix the failure from `{finding.command}` using this evidence:\n{evidence}"
                )
            elif finding.category == "edit_application":
                evidence = finding.evidence[-1600:]
                instructions.append(
                    "Regenerate the smallest edit from the fresh file excerpt below. "
                    "Use a non-empty old_text copied exactly from the current file and include "
                    f"enough context to make it unique:\n{evidence}"
                )
            elif finding.category == "no_changes":
                instructions.append("Produce at least one necessary edit inside the allowed scope.")
            elif finding.category == "scope":
                instructions.append(
                    f"Do not modify `{finding.path}`; keep every edit inside the allowed scope."
                )
        can_retry = repeat_count == 0
        summary = (
            "Repair the reviewer findings and rerun deterministic verification."
            if can_retry
            else "The same deterministic failure repeated; automatic repair is stopped."
        )
        return RepairPlan(
            attempt=len(history) + 1,
            failure_fingerprint=fingerprint,
            summary=summary,
            instructions=instructions,
            failed_commands=failed_commands,
            repeat_count=repeat_count,
            can_retry=can_retry,
        )

    @staticmethod
    def failure_fingerprint(
        review: ReviewDecision, verification: List[CommandResult]
    ) -> str:
        payload = {
            "findings": [
                {
                    "category": finding.category,
                    "command": finding.command,
                    "path": finding.path,
                    "evidence": SelfHealingPlanner._stable_output(finding.evidence),
                }
                for finding in review.findings
            ],
            "verification": [
                {
                    "command": result.command,
                    "exit_code": result.exit_code,
                    "timed_out": result.timed_out,
                    "output": SelfHealingPlanner._stable_output(
                        result.stderr or result.stdout
                    ),
                }
                for result in verification
                if result.exit_code != 0 or result.timed_out
            ],
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
        return sha256(encoded).hexdigest()[:20]

    @staticmethod
    def _stable_output(output: str) -> str:
        stable = output[-1200:]
        stable = re.sub(r"/[^\s:]+/pytest-[^\s:]+", "<pytest-tmp>", stable)
        stable = re.sub(r"\b\d+(?:\.\d+)?s\b", "<duration>", stable)
        stable = re.sub(r"0x[0-9a-fA-F]+", "<address>", stable)
        return stable
