FROM huggingface/transformers-pytorch-gpu
# Install Git and Git LFS
RUN pip install --upgrade huggingface_hub

RUN apt-get update && \
    apt-get install -y git git-lfs && \
    git lfs install
RUN pip install packaging ninja
# Clone and install flash-attention
RUN git clone  https://github.com/Dao-AILab/flash-attention && \
  cd flash-attention && pip install .

# Set environment variables
WORKDIR /submission
COPY . /submission/
# Please put your own huggingface token here that has the write permission
ENV HUGGINGFACE_TOKEN=xxxx
RUN huggingface-cli login --token $HUGGINGFACE_TOKEN
RUN pip install -r requirements.txt
# Install other dependencies
RUN pip install accelerate==0.24.0 transformers==4.34.0 bitsandbytes==0.41.0 tiktoken einops transformers_stream_generator==0.0.4
# Run cache script and start FastAPI
RUN ls
CMD ["python3","train_4090_2.py"]

