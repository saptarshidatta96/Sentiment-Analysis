FROM python:3.7-slim
COPY . /app
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt
EXPOSE 8501
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]