# pythonのDockerfile
FROM python:3.10-slim

ARG DOCKER_WORKDIR=${DOCKER_WORKDIR:-"/app"}
ENV PYTHONUNBUFFERED=1 \
    C_FORCE_ROOT=1

RUN apt update && apt install -y --no-install-recommends \
    build-essential \
    sox \
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
    && rm -rf /var/lib/apt/lists/*

WORKDIR ${DOCKER_WORKDIR}

RUN pip install --no-cache-dir poetry==1.2.* && \
    poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock ./

RUN poetry install

RUN git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git && \
    cd mecab-ipadic-neologd && \
    ./bin/install-mecab-ipadic-neologd -n -y && \
    echo dicdir = `mecab-config --dicdir`"/mecab-ipadic-neologd">/etc/mecabrc && \
    sudo cp /etc/mecabrc /usr/local/etc && \
    cd ..

COPY . .

RUN poetry install
