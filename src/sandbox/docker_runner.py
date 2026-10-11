"""Docker-based isolated execution sandbox."""

import io
import tarfile
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Optional, Tuple
import docker
from docker.models.containers import Container
from docker.errors import DockerException, NotFound, ImageNotFound

from src.config import settings
from src.sandbox.base import BaseSandbox

logger = logging.getLogger(__name__)


class DockerSandboxRunner(BaseSandbox):
    """Secure, resource-constrained Docker sandbox for code execution."""

    def __init__(
        self,
        image: Optional[str] = None,
        mem_limit: Optional[str] = None,
        nano_cpus: Optional[int] = None,
        network_mode: Optional[str] = None,
        user: Optional[str] = None,
    ):
        self.image = image or settings.SANDBOX_IMAGE
        self.mem_limit = mem_limit or settings.SANDBOX_MEM_LIMIT
        self.nano_cpus = nano_cpus or settings.SANDBOX_NANO_CPUS
        self.network_mode = network_mode or settings.SANDBOX_NETWORK_MODE
        self.user = user or settings.SANDBOX_USER

        self.client: Optional[docker.DockerClient] = None
        self.container: Optional[Container] = None

    def _get_client(self) -> docker.DockerClient:
        if self.client is None:
            self.client = docker.from_env()
        return self.client

    def _ensure_image_exists(self) -> str:
        """Resolve available Docker image, falling back if necessary."""
        client = self._get_client()
        try:
            client.images.get(self.image)
            return self.image
        except ImageNotFound:
            logger.info("Image '%s' not found locally. Checking fallback '%s'", self.image, settings.SANDBOX_FALLBACK_IMAGE)
            try:
                client.images.get(settings.SANDBOX_FALLBACK_IMAGE)
                return settings.SANDBOX_FALLBACK_IMAGE
            except ImageNotFound:
                logger.info("Pulling fallback image '%s'...", settings.SANDBOX_FALLBACK_IMAGE)
                client.images.pull(settings.SANDBOX_FALLBACK_IMAGE)
                return settings.SANDBOX_FALLBACK_IMAGE

    def start(self) -> None:
        """Boot a minimal container with strict resource and network isolation."""
        client = self._get_client()
        target_image = self._ensure_image_exists()

        logger.info(
            "Starting Docker sandbox with image=%s, mem_limit=%s, cpus=%s, network=%s, user=%s",
            target_image,
            self.mem_limit,
            self.nano_cpus,
            self.network_mode,
            self.user,
        )

        try:
            self.container = client.containers.run(
                image=target_image,
                command=["/bin/sh", "-c", "tail -f /dev/null"],
                detach=True,
                mem_limit=self.mem_limit,
                nano_cpus=self.nano_cpus,
                network_mode=self.network_mode,
                user=self.user,
                working_dir="/tmp",
                auto_remove=False,
            )
            logger.info("Sandbox container started with ID: %s", self.container.id[:12])
        except DockerException as e:
            logger.error("Failed to start Docker sandbox container: %s", e)
            raise

    def write_file(self, filename: str, content: str, target_dir: str = "/tmp") -> None:
        """Write content into a file inside the container via tar stream archive."""
        if not self.container:
            raise RuntimeError("Sandbox container is not running. Call start() first.")

        data = content.encode("utf-8")
        tar_stream = io.BytesIO()

        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            tarinfo = tarfile.TarInfo(name=filename)
            tarinfo.size = len(data)
            # Permissive mode so unprivileged 'nobody' user can read/write
            tarinfo.mode = 0o666
            tar.addfile(tarinfo, io.BytesIO(data))

        tar_stream.seek(0)
        self.container.put_archive(path=target_dir, data=tar_stream.getvalue())
        logger.debug("Wrote file '%s' (%d bytes) to container directory '%s'", filename, len(data), target_dir)

    def execute(self, cmd: str, timeout: int = 10) -> Tuple[int, str, str]:
        """Execute command in sandbox with strict timeout and output demuxing."""
        if not self.container:
            raise RuntimeError("Sandbox container is not running. Call start() first.")

        logger.debug("Executing in sandbox: %s (timeout=%ds)", cmd, timeout)

        def _run_exec():
            # demux=True separates stdout and stderr streams
            return self.container.exec_run(
                cmd=["/bin/sh", "-c", cmd],
                demux=True,
                workdir="/tmp",
            )

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_exec)
            try:
                exec_result = future.result(timeout=timeout)
                exit_code = exec_result.exit_code
                stdout_bytes, stderr_bytes = exec_result.output or (b"", b"")
                stdout = (stdout_bytes or b"").decode("utf-8", errors="replace")
                stderr = (stderr_bytes or b"").decode("utf-8", errors="replace")
                return exit_code, stdout, stderr
            except TimeoutError:
                logger.warning("Execution timed out after %d seconds: %s", timeout, cmd)
                return -1, "", f"Execution timed out after {timeout} seconds limit."
            except Exception as e:
                logger.error("Error executing command in sandbox: %s", e)
                return -1, "", str(e)

    def cleanup(self) -> None:
        """Gracefully stop and remove the sandbox container."""
        if self.container:
            container_id = self.container.id[:12]
            try:
                logger.info("Stopping sandbox container %s...", container_id)
                self.container.stop(timeout=2)
            except (NotFound, DockerException) as e:
                logger.debug("Sandbox container %s stop warning: %s", container_id, e)

            try:
                logger.info("Removing sandbox container %s...", container_id)
                self.container.remove(force=True)
            except (NotFound, DockerException) as e:
                logger.debug("Sandbox container %s remove warning: %s", container_id, e)

            self.container = None
            logger.info("Sandbox container %s cleaned up.", container_id)
