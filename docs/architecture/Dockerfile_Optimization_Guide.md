# Dockerfile Optimization Guide

This guide provides best practices for designing and writing Dockerfiles to create optimized, efficient, and secure Docker images. The goal is to minimise image size, reduce build times, and ensure that images are suitable for use in development, testing (including GitHub Actions), and production environments.

## Key Principles

- **Minimise Image Size:** Smaller images are faster to pull, push, and deploy.
- **Leverage Build Cache:** Structure your Dockerfile to take advantage of Docker's build cache for faster builds.
- **Create Reproducible Builds:** Dockerfiles should produce the same image every time they are built.
- **Keep Images Secure:** Follow security best practices to reduce the attack surface of your containers.

## Best Practices

### 1. Use a Specific Base Image

Always use a specific tag for your base image (e.g., `python:3.10-slim-bullseye` instead of `python:latest`). This ensures that your builds are reproducible and that you are not accidentally introducing breaking changes from upstream images.

For easier maintenance, especially in projects with multiple Dockerfiles or complex multi-stage builds, you can use an `ARG` to define the base image. This allows you to specify the base image version in one place or even pass it as a build-time argument.

**Example with `ARG`:**
```dockerfile
ARG BASE_IMAGE=python:3.10-slim-bullseye

# ---- Build Stage ----
FROM ${BASE_IMAGE} as builder

# ... build steps ...

# ---- Final Stage ----
FROM ${BASE_IMAGE}

# ... final image steps ...
```
You can then build with a custom base image like so:
`docker build --build-arg BASE_IMAGE=python:3.11-slim-bullseye -t my-app .`

### 2. Multi-Stage Builds

Use multi-stage builds to separate the build environment from the runtime environment. This is one of the most effective ways to reduce the size of your final image.

**Example:**

```dockerfile
# ---- Build Stage ----
FROM python:3.10 as builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ---- Final Stage ----
FROM python:3.10-slim-bullseye

WORKDIR /app

COPY --from=builder /app /app

CMD ["python", "app/main.py"]
```

### 3. Minimise Layers

Each `RUN`, `COPY`, and `ADD` instruction in a Dockerfile creates a new layer. To minimise the number of layers, chain related commands together using the `&&` operator.

**Good:**
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
 && rm -rf /var/lib/apt/lists/*
```

**Bad:**
```dockerfile
RUN apt-get update
RUN apt-get install -y --no-install-recommends gcc
RUN apt-get install -y --no-install-recommends g++
RUN rm -rf /var/lib/apt/lists/*
```

### 4. Optimize `COPY` and `ADD`

- Be specific with `COPY` commands. Only copy what you need.
- Order `COPY` commands from least to most frequently changing files. For example, copy `requirements.txt` or `package.json` and install dependencies *before* copying the rest of the application code. This allows you to leverage the build cache more effectively.

### 5. Use a `.dockerignore` File

Create a `.dockerignore` file to exclude files and directories that are not needed in the final image. This can significantly reduce the size of the build context and the final image.

**Example `.dockerignore`:**
```
.git
.gitignore
.vscode
.idea
*.pyc
__pycache__/
node_modules/
.venv/
```

### 6. Clean Up After Yourself

In `RUN` commands, always clean up temporary files and caches in the same layer. For example, remove package manager caches after installing packages.

## Dockerfiles for GitHub Actions

When running tests in GitHub Actions, you can use the same Dockerfile as you would for production, or you can create a dedicated Dockerfile for testing. A testing Dockerfile might include additional dependencies, such as testing frameworks and linters.

By following these guidelines, you can create Docker images that are small, fast, and secure, and that are well-suited for use in a CI/CD pipeline with GitHub Actions.
