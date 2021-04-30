
# start with a cuda base image
FROM nvidia/cuda:10.1-runtime-ubuntu18.04

# install packages and python
RUN apt-get update && \
    apt-get install python3.6 python3-pip -y  &&\
    apt-get install gcc libev-dev libffi-dev -y && \
    apt-get clean

# update pip to install dependencies
RUN pip3 install --upgrade pip

# Copy the requirements and install
WORKDIR /app
COPY requirements.txt /app
RUN pip3 install -r requirements.txt

# Run the special pytorch download
RUN pip3 install torch==1.7.1+cu101 torchvision==0.8.2+cu101 \
 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install kubeml==0.1.6


# Copy the server and start
COPY . /app

# Start a gunicorn server with only one worker to ensure
# isolation on a function basis
# and offer more performance than the default server
CMD ["gunicorn", "-b", "0.0.0.0:8888", "-t", "2000", "-w 1", "server:app"]

