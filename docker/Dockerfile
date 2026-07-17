# RouteLLM + NotDiamond router dry-run — Linux so the ML stack actually installs.
#
# The whole reason this exists: RouteLLM's litellm/torch/transformers/datasets
# stack has no working install path on the Windows host (no Rust toolchain, wheel
# gaps, version tangles). On a Linux base it's all manylinux wheels. The host is
# left untouched; this image is the only place RouteLLM lives.
#
# It's a BATCH job, not a server: it runs run_router_dryrun.py over the 45
# questions and writes results/router_selections.csv to the mounted repo. No ports.
FROM python:3.11-slim

WORKDIR /app

# git + a compiler in case any dep lacks a wheel and falls back to a source build.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

# litellm<1.90 stays pure-Python (avoids the 1.92 Rust build). routellm pulls the
# ML stack; notdiamond is the light API client; pandas is used by both routellm
# and the dry-run. Pin here if the resolver ever picks a conflicting set.
RUN pip install --no-cache-dir \
    "litellm<1.90" \
    routellm \
    notdiamond \
    python-dotenv \
    pandas

# Separate layer (keeps the big ML layer cached): anthropic is only needed to
# generate candidate-model responses for the NotDiamond custom-router TRAINING set.
RUN pip install --no-cache-dir anthropic

# Code + data are mounted at runtime (see docker-compose.yml), not copied, so the
# image is just the environment and results land back on the host.
CMD ["python", "run_router_dryrun.py"]
