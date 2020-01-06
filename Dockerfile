FROM nvcr.io/nvidia/tensorflow:18.06-py3

RUN apt update && apt install -y libsm6 libxext6 libxrender-dev && \
    pip install -r requirements.txt

WORKDIR /home/kensert_CNN

COPY *.py ./
COPY LICENSE .
#COPY bbbc014_labels.npy .

RUN mkdir images_bbbc014 && mkdir images_bbbc021 && mkdir BBBC014_v1_images && mkdir act_max_output

COPY requirements.txt .
