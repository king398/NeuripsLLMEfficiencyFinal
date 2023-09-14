FROM pytorch/pytorch
RUN cd ..
COPY inference /submission
WORKDIR /submission
CMD ["python3", "main.py"]
