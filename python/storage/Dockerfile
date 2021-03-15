# Builder image
FROM python:3.6-slim-buster as builder

RUN mkdir -p /app
RUN mkdir -p /app/uploads
WORKDIR /app

COPY requirements.txt /app
RUN pip3 install  -r requirements.txt

COPY  api.py /app
COPY  utils.py /app


#ENV DEBUG_ENV True

EXPOSE 9090

CMD ["gunicorn", "-b", "0.0.0.0:9090", "-t", "200", "-w 4", "api:app"]