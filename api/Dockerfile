FROM python:3.10-slim

WORKDIR /app

# Copy only necessary files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y locales
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

COPY . .

# Verify the presence of the model files
RUN ls -l models/

CMD ["uvicorn", "utils.preprocess:app", "--host", "0.0.0.0", "--port", "8000"]