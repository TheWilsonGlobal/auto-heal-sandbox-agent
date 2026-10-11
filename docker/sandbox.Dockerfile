# Minimal, hardened execution sandbox for code execution
FROM python:3.11-slim

# Pre-install pytest and common testing utilities
RUN pip install --no-cache-dir pytest

# Create and grant permissions on working directory
WORKDIR /tmp
RUN chmod 1777 /tmp

# Default to unprivileged nobody user for strict security
USER nobody

CMD ["tail", "-f", "/dev/null"]
