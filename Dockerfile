# Use a Python version compatible with your dependencies (e.g., 3.9, 3.10, 3.11)
FROM python:3.11

# Set up a non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
ENV HOME=/home/user

WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application code and data
COPY --chown=user . /app

# Expose the port HF Spaces expects
EXPOSE 7860

# Command to run the application using uvicorn
# It looks for the 'app' instance inside the 'api.py' file
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]