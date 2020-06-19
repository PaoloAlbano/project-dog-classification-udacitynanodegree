FROM python:3.6-slim-stretch

RUN apt-get update
RUN apt-get -y install build-essential gcc python3-dev python3-pip
RUN pip install --upgrade pip

ADD requirements.txt /dog/
WORKDIR /dog
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

ADD web_app/* /dog/
ADD model_transfer.pt /dog/

CMD ["gunicorn","-w","1","-b","0.0.0.0:8080","-t", "1000","server:app"]
EXPOSE 8080



