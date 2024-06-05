# syntax = docker/dockerfile:1.4
FROM python:3.10

# Install system dependencies
RUN --mount=type=cache,target=/var/cache/apt \
  apt-get update && \
  apt-get install -y --no-install-recommends net-tools nano

ENV PYTHONDONTWRITEBYTECODE=1

# Create and activate the omost environment
RUN python -m venv /opt/omost && \
  /opt/omost/bin/pip install --upgrade pip

# Set the PATH to prioritize the virtualenv
ENV PATH="/opt/omost/bin:$PATH"

# Install necessary packages
WORKDIR /app

# Install additional requirements based on the target platform with apk package caching
ARG TARGET=cuda
RUN --mount=type=cache,target=/root/.cache/pip \
  if [ "$TARGET" = "cuda" ]; then \
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121; \
  elif [ "$TARGET" = "rocm" ]; then \
  pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.2; \
  else \
  pip install torch torchvision; \
  fi

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

COPY . .

ENV GRADIO_SERVER_PORT="7860"
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV PYTHONUNBUFFERED=1

USER 33:33

VOLUME [ "/app/hf_download" ]

EXPOSE 7860
CMD ["python", "-u", "gradio_app.py"]
