FROM python:3.11.1

WORKDIR .

COPY . .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python","./api_titanic.py"]