ARG BASE_IMAGE=python:3.9-alpine 
FROM ${BASE_IMAGE} AS echo_api_builder

WORKDIR /build

# Install build dependencies for Alpine
RUN apk add --no-cache \
	gcc \
	g++ \
	musl-dev \
	linux-headers \
	libffi-dev \
	openssl-dev \
	python3-dev \
	make 

# Create virtual environment
RUN python -m venv /opt/venv 
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 


# Runtime stage - slim final image
FROM ${BASE_IMAGE}

WORKDIR /app

# Install only runtime dependencies (not build tools)
RUN apk add --no-cache \
	libstdc++ \
	libgcc

COPY --from=echo_api_builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY . /app

EXPOSE 8080

# Start the API using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
