# 🏗️ Use an official lightweight Python image
FROM python:3.11-slim

# 📁 Set the working directory in the container
WORKDIR /app

# 🔥 Copy the FastAPI application and models
COPY app/ app/
COPY models/ models/

# 📦 Install dependencies
RUN pip install --no-cache-dir -r app/requirements.txt

# 🌎 Expose the FastAPI port
EXPOSE 8000

# 🚀 Start the FastAPI application
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
