FROM huggingface/transformers-pytorch-gpu
RUN cd ..
COPY . /submission
WORKDIR /submission
RUN pip install  accelerate
RUN pip install -i https://test.pypi.org/simple/ bitsandbytes
RUN accelerate test
RUN pip install -r requirements.txt
ENV HUGGINGFACE_TOKEN=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
CMD ["python3","-m","uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]