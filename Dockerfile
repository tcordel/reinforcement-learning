FROM python:latest

WORKDIR /data
RUN pip install numpy
RUN pip install tensorboard
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
CMD ["python3", "uttt_ppo_self_play.py"]
