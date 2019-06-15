FROM ubuntu:18.04

# Install: vim, git, cron, wget
RUN apt-get update && apt-get -y install apt-file && apt-file update && \
    apt-get -y install cron vim git wget sudo gcc curl

# place to keep our app and the data:
RUN mkdir -p /app && mkdir -p /app/logs && mkdir -p /data && mkdir -p /_tmp

WORKDIR /app

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
COPY braai/requirements.cpu.txt /app/
RUN pip install -r /app/requirements.cpu.txt
COPY braai/ /app/

# copy over the secrets:
#COPY secrets.json /app/

CMD /bin/bash