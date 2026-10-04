"""SmartOS — chat-driven OS management by sandboxed agents."""

from ..features import require_smartos

require_smartos()  # switched off -> SmartOSDisabledError (see lightagentx.features)

from .apps import make_app_tools
from .files import make_file_tools
from .os_agent import SmartOS, build_os_tools
from .security import make_security_tools
from .shell import make_shell_tools
from .tasks import make_task_tools

__all__ = [
    "SmartOS", "build_os_tools",
    "make_file_tools", "make_task_tools", "make_app_tools",
    "make_security_tools", "make_shell_tools",
]
