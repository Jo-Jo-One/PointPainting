FROM nvidia/cuda:11.3.1-devel-ubuntu20.04
LABEL authors="jeongmin"


RUN apt-get update && apt-get install wget -yq
RUN apt-get install build-essential g++ gcc -y
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get install libgl1-mesa-glx libglib2.0-0 -y
RUN apt-get install libgtk2.0-dev git -y

ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda

ENV PATH=$CONDA_DIR/bin:$PATH

RUN git clone https://github.com/open-mmlab/mmsegmentation.git /mmsegmentation
RUN conda install --file spec-file.txt