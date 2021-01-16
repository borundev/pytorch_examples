FROM nvidia/cuda:11.1-base-ubuntu20.04
RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y wget


RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda
ENV PATH=/miniconda/bin:$PATH


COPY environment.yml environment.yml
COPY requirements.txt requirements.txt
RUN conda env update -f environment.yml
RUN pip install torchsummary
RUN pip install plotly
