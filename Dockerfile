FROM python:3.12-slim

WORKDIR /app

COPY ./requirements.txt /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache -r requirements.txt

COPY ./src ./src
COPY ./settings.py .
COPY ./find_cameras.py .
COPY ./run.py .

EXPOSE 5000 5001

CMD ["python", "run.py"]
