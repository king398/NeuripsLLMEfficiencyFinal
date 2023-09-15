FROM pytorch/pytorch
RUN cd ..
COPY . /submission
WORKDIR /submission
RUN pip install bitsandbytes
RUN pip install -r requirements.txt
CMD ["python3", "inference/main.py"]
