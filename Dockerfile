FROM python:3.10

WORKDIR /app

COPY app/ /app

RUN pip install flask joblib tensorflow keras pillow librosa numpy matplotlib

EXPOSE 5000

CMD ["python", "app.py"]

