FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:/usr/local/bin:${PATH}"

WORKDIR /app

# Hermes installer prerequisites: git, curl, xz-utils (Node.js ships as .tar.xz)
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl git xz-utils ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Install Hermes Agent from the OFFICIAL Nous Research installer.
# No "|| true": if the install fails, the build must fail too.
RUN curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh \
      -o /tmp/hermes-install.sh \
 && bash /tmp/hermes-install.sh --skip-browser \
 && rm /tmp/hermes-install.sh \
 && (command -v hermes || ln -s /root/.local/bin/hermes /usr/local/bin/hermes) \
 && hermes --version

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "-m", "backend.web"]
