from __future__ import annotations

import argparse
import json
import sys

from app.services.consistency_checker import apply_safe_fixes, run_consistency_checks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report or safely fix Project Echo database consistency issues.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Print a report without changing the database (default).",
    )
    parser.add_argument(
        "--safe-fix",
        action="store_true",
        help="Apply conservative fixes for C14.2-SEED fixture records only.",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required with --safe-fix before any database changes are made.",
    )
    args = parser.parse_args()

    issues = run_consistency_checks()

    if args.safe_fix:
        if not args.confirm:
            print(
                json.dumps(
                    {
                        "error": "Refusing safe-fix without --confirm",
                        "issues_found": len(issues),
                    },
                    indent=2,
                )
            )
            return 2

        fixed = apply_safe_fixes(issues)
        print(
            json.dumps(
                {
                    "mode": "safe-fix",
                    "issues_found": len(issues),
                    "fixed": fixed,
                },
                indent=2,
            )
        )
        return 0

    print(
        json.dumps(
            {
                "mode": "report-only",
                "issues_found": len(issues),
                "issues": issues,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
