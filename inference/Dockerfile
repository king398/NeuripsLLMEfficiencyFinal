FROM huggingface/transformers-pytorch-gpu

# Install Git and Git LFS
RUN apt-get update && \
    apt-get install -y git git-lfs && \
    git lfs install

# Set environment variables
ENV HUGGINGFACE_TOKEN=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

# Install Hugging Face CLI
RUN pip install huggingface_hub

# Log in to Hugging Face
RUN huggingface-cli login --token $HUGGINGFACE_TOKEN

# Download the models using Hugging Face CLI
RUN huggingface-cli  download meta-llama/Llama-2-13b-hf --exclude="*.bin"
RUN huggingface-cli  download Mithilss/Llama-2-13b-hf-2-epoch-checkpoint-10044

# Set working directory and copy files
WORKDIR /submission
COPY . /submission

# Install other dependencies
RUN pip install git+https://github.com/huggingface/accelerate.git
RUN pip install -i https://test.pypi.org/simple/ bitsandbytes
RUN pip install -r requirements.txt

# Run cache script and start FastAPI
RUN ls
CMD ["python3","-m","uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
