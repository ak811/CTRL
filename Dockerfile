# CPU image for reproducible experiments.
#   docker build -t ctrl .
#   docker run --rm -v "$PWD/outputs:/app/outputs" ctrl python -m continual.run
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md LICENSE ./
COPY common ./common
COPY envs ./envs
COPY transfer ./transfer
COPY meta ./meta
COPY continual ./continual
RUN pip install -e ".[atari,tensorboard,dev]"
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests

CMD ["bash", "scripts/smoke_test.sh"]
