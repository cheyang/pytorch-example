FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

RUN mkdir /root/mnist-pytorch

ADD . /root/mnist-pytorch

RUN pip install tensorboard && chmod u+x /root/mnist-pytorch/run.sh

