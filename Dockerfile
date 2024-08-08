FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

#RUN pip install --upgrade keras

COPY . .

EXPOSE 80

#CMD ["gunicorn", "--bind", "0.0.0.0:5000", "server:app"]
#CMD ["gunicorn", "--bind", "0.0.0.0:80", "app"]
CMD ["python", "app.py"]