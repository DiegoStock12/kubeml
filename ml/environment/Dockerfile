
# Builder image
FROM python:3.6-slim-buster as builder

# Update the packages
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install gcc libev-dev libffi-dev -y && \
    apt-get clean


RUN pip3 install --upgrade pip

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

# nvidia gpu stuff
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Start a gunicorn server with only one worker to ensure
# isolation on a function basis
# and offer more performance than the default server
CMD ["gunicorn", "-b", "0.0.0.0:8888", "-t", "2000", "-w 1", "server:app"]

#ENTRYPOINT ["python3"]
#CMD ["server.py"]

