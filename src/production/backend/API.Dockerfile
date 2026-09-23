ARG BASE_IMAGE=python:3.9-alpine
FROM ${BASE_IMAGE} AS echo_api_builder

WORKDIR /build

RUN apk add --no-cache \
        gcc \
        g++ \
        musl-dev \
        linux-headers \
        libffi-dev \
        openssl-dev \
        python3-dev \
        make

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


FROM ${BASE_IMAGE}

WORKDIR /app

RUN apk add --no-cache \
        libstdc++ \
        libgcc

COPY --from=echo_api_builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY backend/ /app
COPY infrastructure/store/cloudflare_r2 \
    /opt/project_echo_shared/cloudflare_r2

ENV PYTHONPATH=/opt/project_echo_shared

EXPOSE 9000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000"]
