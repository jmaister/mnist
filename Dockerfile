
# docker build -t mnist .
# docker build --build-arg HTTP_PROXY=http://bluecoat-proxy:8080 --build-arg HTTPS_PROXY=http://bluecoat-proxy:8080 -t mnist .
# docker run -v c:\dev\workspace\mnist:/root/mnist -v c:\dev\workspace\mnist\output:/output -it mnist bash

# https://hub.docker.com/r/waleedka/modern-deep-learning/

#FROM ubuntu:16.04
#FROM floydhub/dl-docker:cpu
#FROM waleedka/modern-deep-learning
FROM python:3.6.5

RUN apt-get update -y && apt-get upgrade -y

RUN pip install --upgrade pip

RUN apt-get install nano -y

WORKDIR /root

ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt -v

RUN python -c "import mnist; mnist.train_images()"

CMD bash

# ADD keras.json /root/.keras/keras.json
