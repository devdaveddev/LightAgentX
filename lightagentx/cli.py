"""`lightx` — umbrella command for LightAgentX tools."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lightx", description="LightAgentX command-line tools.")
    sub = parser.add_subparsers(dest="command", required=True)

    from .doctor.cli import add_arguments as doctor_args
    doctor = sub.add_parser("doctor", help="Find version problems; fix them with verified, approved patches.",
                            description="Find version problems; fix them with verified, approved patches.")
    doctor_args(doctor)

    args = parser.parse_args(argv)
    if args.command == "doctor":
        from .doctor.cli import run
        return run(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
