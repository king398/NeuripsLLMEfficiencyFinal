FROM huggingface/transformers-pytorch-gpu
# Install Git and Git LFS
RUN pip install --upgrade huggingface_hub

RUN apt-get update && \
    apt-get install -y git git-lfs && \
    git lfs install
#RUN pip install packaging ninja
# Clone and install flash-attention
#RUN git clone -b v1.0.8 https://github.com/Dao-AILab/flash-attention && \
#    cd flash-attention && pip install . && pip install csrc/layer_norm csrc/rotary

# Set environment variables
WORKDIR /submission
COPY . /submission/
ENV HUGGINGFACE_TOKEN=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
RUN huggingface-cli login --token $HUGGINGFACE_TOKEN
RUN huggingface-cli  download mistralai/Mistral-7B-v0.1
RUN huggingface-cli download Mithilss/Mistral-7B-1-epoch-platypus-cnn-all-modules
RUN pip install -r requirements.txt
# Install other dependencies
RUN pip install git+https://github.com/huggingface/accelerate.git git+https://github.com/huggingface/transformers.git bitsandbytes transformers_stream_generator==0.0.4 tiktoken

# Run cache script and start FastAPI
RUN ls
CMD ["python3","-m","uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
