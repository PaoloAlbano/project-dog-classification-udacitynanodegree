FROM python:3.6-slim-stretch

RUN apt-get update
RUN apt-get install -y libglib2.0-0
#RUN apt-get -y install build-essential python3-dev python3-pip
#RUN pip install --upgrade pip

RUN pip install gunicorn
RUN pip install falcon==2.0.0
RUN pip install falcon_multipart==0.2.0

ADD requirements.txt /dog/
WORKDIR /dog
RUN mkdir -p images/
RUN pip install --no-cache-dir -r requirements.txt

ADD web_app/* /dog/
ADD model_transfer.pt /dog/

RUN python download_vgg16.py

CMD ["gunicorn","-w","1","-b","0.0.0.0:8080","-t", "1000","server:app"]
EXPOSE 8080



