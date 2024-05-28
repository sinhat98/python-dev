FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ARG DPCKER_WORKDIR /root/workspace

# Install necessary system packages including git
RUN apt update && apt install -y --no-install-recommends \
    python3-pip \
    build-essential \
    openssh-client \
    sox \
    flac \
    nkf \
    libsndfile1 \
    ffmpeg \
    portaudio19-dev \
    wget \
    curl \
    make \
    file \
    python3-pyaudio \
    git \
    libmecab-dev \
    mecab \
    mecab-ipadic-utf8 \
    sudo \
    && apt clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

RUN mkdir -m 700 $HOME/.ssh && ssh-keyscan github.com > $HOME/.ssh/known_hosts

WORKDIR ${DOCKER_WORKDIR}

RUN pip install --no-cache-dir poetry==1.2.* && \
    poetry config virtualenvs.create false


COPY pyproject.toml poetry.lock ./

RUN --mount=type=ssh poetry install --no-root
