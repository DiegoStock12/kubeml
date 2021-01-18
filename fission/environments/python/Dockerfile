
# Builder image
FROM python:3.6-slim-buster as builder

# Update the packages
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install gcc libev-dev libffi-dev -y && \
    apt-get clean


RUN pip3 install --upgrade pip3

# Copy the requirements and install
WORKDIR /app
COPY requirements.txt /app
RUN pip3 install --user -r requirements.txt

# Copy the server and start
COPY . /app


# Production smaller image
FROM python:3.6-slim-buster as final-image

# Install the libraries used by bjoern
RUN apt-get update && \
    apt-get install libev-dev libffi-dev -y

COPY --from=builder /root/.local /root/.local
COPY --from=builder /app/* /app/
WORKDIR app
ENV PATH=/root/.local/bin:$PATH

ENTRYPOINT ["python3"]
CMD ["server.py"]

