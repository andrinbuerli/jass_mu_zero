FROM tensorflow/tensorflow:2.7.0-gpu

ARG DEV

WORKDIR /tmp

RUN apt-get install wget -y

RUN apt-key del 7fa2af80

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

RUN apt-get update && apt-get install build-essential wget git -y

RUN curl -sSL https://cmake.org/files/v3.5/cmake-3.5.2-Linux-x86_64.tar.gz | tar -xzC /opt\
    && apt-get update\
    && wget http://www.cmake.org/files/v3.5/cmake-3.5.2.tar.gz\
    && tar xf cmake-3.5.2.tar.gz\
    && cd cmake-3.5.2\
    &&./configure\
    && make\
    && make install

# RUN apt install libeigen3-dev

RUN git clone https://gitlab.com/libeigen/eigen.git && cd eigen && git checkout 3.4.0
RUN cmake eigen && make install
RUN ln -s /usr/local/include/eigen3/Eigen /usr/local/include/Eigen

RUN wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-2.6.0.tar.gz
RUN tar -C /usr/local -xzf libtensorflow-gpu-linux-x86_64-2.6.0.tar.gz
RUN rm libtensorflow-gpu*

COPY . /app/
WORKDIR /app

# add user
RUN chown -hR 1000 /app
RUN chown -hR 1000 /usr
RUN adduser user --uid 1000
RUN adduser user sudo
USER user

RUN git config --global --add safe.directory /app

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

RUN pip install -v -e .

RUN pip install -r requirements-dev.txt

RUN chmod +x /app/entrypoint.sh

ENV XLA_PYTHON_CLIENT_MEM_FRACTION=.7

ENTRYPOINT ["sh", "/app/entrypoint.sh"]