FROM python:3.10-slim

WORKDIR /app

# Copy only necessary files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Verify the presence of the model files
RUN ls -l models/

CMD ["uvicorn", "utils.preprocess:app", "--host", "0.0.0.0", "--port", "8000"]