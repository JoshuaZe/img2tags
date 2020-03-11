FROM python:3.6-slim

COPY . .

RUN pip install -r requirements.txt

CMD ["python3", "app.py"]
