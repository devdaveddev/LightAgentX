"""lightx doctor — find version problems, and fix them with verified, approved patches."""

from .checks import CheckContext, run_checks, snapshot_layout
from .fixer import Proposal, propose_code_fix, run_fixes
from .migrate import migrate_snapshot_data
from .project import Finding, Workspace

__all__ = ["Finding", "Workspace", "CheckContext", "run_checks", "run_fixes",
           "propose_code_fix", "Proposal", "migrate_snapshot_data", "snapshot_layout"]
