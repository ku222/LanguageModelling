

FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

WORKDIR /model

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD ["python", "train.py"]
