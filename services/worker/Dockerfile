FROM python:3.13.7

# Set the working directory in the container
WORKDIR /usr/src/app

# Set environment variables for a non-interactive build
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV POETRY_NO_INTERACTION=1

# Install Poetry package manager
RUN pip install poetry

# Copy only the dependency files to leverage Docker layer caching
COPY pyproject.toml ./

# Generate lock file and install project dependencies
RUN poetry lock && poetry install --no-root --only=main

# Copy the rest of the application code
COPY . .

# Expose health check port
EXPOSE 8000

# Command to run the worker
CMD ["poetry", "run", "python", "main.py"]