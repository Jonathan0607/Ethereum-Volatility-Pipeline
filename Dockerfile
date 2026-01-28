FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. THE WORKSPACE
# "Create a folder called /app and stand inside it."
WORKDIR /app

# 4. THE DEPENDENCIES
# "Update the Linux package manager (apt-get) and install build tools."
# We need 'build-essential' because some math libraries (like ARCH) need C++ compilers.
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. THE LIBRARIES
# "Copy the grocery list (requirements.txt) into the container."
# "Install them."
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. THE CODE
# "Copy everything from your laptop's current folder into the container's /app folder."
COPY . .

# 7. THE ACTION
# "When someone turns this container on, run this specific command."
CMD ["python", "src/pipeline.py"]