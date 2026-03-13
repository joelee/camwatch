FROM python:3.14-bookworm

RUN mkdir /config
RUN mkdir /recordings
RUN mkdir -p /opt/camwatch/src /opt/camwatch/config

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:${PATH}"

COPY pyproject.toml README.md requirements.txt /opt/camwatch/
COPY src /opt/camwatch/src/
COPY config/camwatch-quick_start.yaml /config/camwatch.yaml
COPY config/camwatch-defaults.yaml /config/camwatch-defaults.yaml

RUN uv sync --project /opt/camwatch

RUN groupadd -g 1001 camwatch
RUN useradd -u 1001 -g camwatch -m -s /bin/bash camwatch
RUN chown -R camwatch:camwatch /recordings
RUN chown -R camwatch:camwatch /opt/camwatch

ENV CV_CONFIG_PATH /config

USER camwatch
WORKDIR /opt/camwatch

CMD ["bash"]
# CMD ["python", "./main.py"]
