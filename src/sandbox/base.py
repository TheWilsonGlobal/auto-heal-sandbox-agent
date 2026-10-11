"""Base interface definition for execution sandboxes."""

from abc import ABC, abstractmethod
from typing import Tuple


class BaseSandbox(ABC):
    """Abstract interface for isolated execution environments."""

    @abstractmethod
    def start(self) -> None:
        """Initialize and start the sandbox environment."""
        pass

    @abstractmethod
    def write_file(self, filename: str, content: str, target_dir: str = "/tmp") -> None:
        """Write content into a specified file inside the sandbox.

        Args:
            filename: Name of the file (e.g. 'solution.py')
            content: Raw string content of the file
            target_dir: Directory inside sandbox where the file is written
        """
        pass

    @abstractmethod
    def execute(self, cmd: str, timeout: int = 10) -> Tuple[int, str, str]:
        """Execute a shell command inside the sandbox.

        Args:
            cmd: Command string to execute
            timeout: Maximum execution timeout in seconds

        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Stop and tear down sandbox resources."""
        pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
