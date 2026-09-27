# The base image pins the CUDA runtime the reported configuration was declared for, and the Python
# version matches the one the release is tested against.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/prism/code

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/prism

COPY requirements.txt ./
RUN python3.11 -m pip install --no-cache-dir --upgrade pip \
    && python3.11 -m pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.txt ./
COPY code ./code
COPY configs ./configs
COPY scripts ./scripts

RUN python3.11 -m pip install --no-cache-dir -e .

RUN useradd --create-home --shell /bin/bash prism
USER prism

ENTRYPOINT ["python3.11", "-m", "prism.cli"]
CMD ["audit"]
