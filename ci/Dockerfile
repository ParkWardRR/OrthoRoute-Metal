# =============================================================================
# OrthoRoute-Metal CI Container
# For use with OrbStack (https://orbstack.dev) or Docker
# =============================================================================

FROM python:3.12-slim

LABEL maintainer="OrthoRoute Team"
LABEL description="CI environment for OrthoRoute-Metal"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python deps (excluding GPU and KiCad-specific packages)
WORKDIR /workspace
COPY requirements.txt .

# Install only CI-compatible deps (skip cupy, kicad-python)
RUN pip install --no-cache-dir \
    numpy>=1.20.0 \
    psutil>=5.8.0 \
    pytest>=6.0.0 \
    pytest-cov>=2.0.0 \
    flake8>=3.8.0 \
    mypy>=0.800 \
    black>=21.0.0

# Copy project (volume-mounted at runtime, this is just for image layer caching)
COPY . .

# Default: run tests
CMD ["pytest", "tests/", "-v", "--tb=short"]
