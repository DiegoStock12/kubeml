# Builder image
FROM python:3.6-slim-buster as builder

RUN mkdir -p /app
WORKDIR /app

COPY requirements.txt /app
RUN pip3 install  -r requirements.txt

COPY  api.py /app
COPY  utils.py /app

EXPOSE 80

CMD ["gunicorn", "-b", "0.0.0.0:80", "-w 1", "api:app"]