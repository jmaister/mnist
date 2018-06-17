
# docker build -t burgbot .
# docker build --build-arg HTTP_PROXY=http://bluecoat-proxy:8080 --build-arg HTTPS_PROXY=http://bluecoat-proxy:8080 -t burgbot .
# docker run -v c:\dev\workspace\numerai:/root/numerai -v c:\dev\workspace\numerai\output:/output -it burgbot bash

# https://hub.docker.com/r/waleedka/modern-deep-learning/

#FROM floydhub/dl-docker:cpu
FROM waleedka/modern-deep-learning

RUN apt-get update -y && apt-get upgrade -y

RUN pip install --upgrade pip
RUN pip install --ignore-installed --upgrade "https://github.com/mind/wheels/releases/download/tf1.6-cpu/tensorflow-1.6.0-cp35-cp35m-linux_x86_64.whl"

RUN apt-get install nano -y


WORKDIR /root

COPY numerai_datasets.zip numerai_datasets.zip
RUN unzip numerai_datasets.zip -d /data && rm numerai_datasets.zip

#ADD requirements.txt requirements.txt
#RUN pip install -r requirements.txt

#RUN pip3 install --no-cache-dir --upgrade pydot graphviz

CMD bash
#CMD python train.py

# ADD keras.json /root/.keras/keras.json
