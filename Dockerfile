FROM python:3.9

WORKDIR /app

COPY . .

RUN pip install flask

EXPOSE 8080

CMD ["python", "inference.py"]