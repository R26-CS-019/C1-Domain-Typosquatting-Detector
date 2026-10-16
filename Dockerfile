FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONDONTWRITEBYTECODE=1
WORKDIR /home/user/app

COPY --chown=user . .
RUN pip install --no-cache-dir --user -r requirements_api.txt

EXPOSE 7860
CMD ["python", "-m", "uvicorn", "integration.domain_typosquatting_api:app", "--host", "0.0.0.0", "--port", "7860"]