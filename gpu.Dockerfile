FROM nvidia/cuda:10.0-devel-ubuntu18.04

# Install: vim, git, cron, wget
RUN apt-get update && apt-get -y install apt-file && apt-file update && \
    apt-get -y install cron vim git wget sudo gcc curl && \
    apt-get -y install gnupg2

# place to keep our app and the data:
RUN mkdir -p /app && mkdir -p /app/logs && mkdir -p /data && mkdir -p /_tmp

WORKDIR /app

ENV CUDNN_VERSION 7.5.0.56

RUN apt-get update && apt-get install -y --no-install-recommends \
            libcudnn7=$CUDNN_VERSION-1+cuda10.0 \
            libcudnn7-dev=$CUDNN_VERSION-1+cuda10.0 && \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*


# Install TensorRT. Requires that libcudnn7 is installed above.
#RUN apt-get update && \
#    apt-get install -y nvinfer-runtime-trt-repo-ubuntu1804-5.0.2-ga-cuda10.0 && \
#    apt-get update && \
#    apt-get install -y --no-install-recommends libnvinfer-dev=5.0.2-1+cuda10.0

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user && \
    chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

RUN sudo chown -R user:users /data /app /_tmp

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install latest Miniconda
RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p ~/miniconda
ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# copy code over and install python libs
COPY braai/requirements.gpu.txt /app/
RUN pip install -r /app/requirements.gpu.txt
RUN pip install tensorflow-gpu==2.0.0b0
COPY braai/ /app/

# copy over the secrets:
#COPY secrets.json /app/

CMD /bin/bash