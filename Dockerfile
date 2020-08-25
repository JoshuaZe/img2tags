FROM python:3.6-slim

COPY img2tags             img2tags
COPY test                 test
COPY models               models
COPY app.py               app.py
COPY vocabulary.py        vocabulary.py
COPY requirements.txt     requirements.txt

RUN pip install -r requirements.txt

CMD ["python3", "app.py"]
