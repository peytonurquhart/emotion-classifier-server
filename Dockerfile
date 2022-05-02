FROM ubuntu:latest
ENV DEBIAN_FRONTEND=noninteractive
MAINTAINER Ilya Pukhov "ilya.pukhov@gmail.com"
RUN apt-get update -y
RUN apt-get install -y python3-pip \
    python3-dev \
    build-essential \
    cmake
COPY . /app
WORKDIR /app
EXPOSE 8080
RUN pip3 install flask
RUN pip3 install dlib
RUN pip3 install flask_cors
RUN pip3 install face_recognition
RUN pip3 install face_recognition_models
RUN pip3 install joblib
RUN pip3 install matplotlib
RUN pip3 install numpy
RUN pip3 install scikit-image
RUN pip3 install sklearn
RUN pip3 install scipy
RUN pip3 install imageio
ENTRYPOINT ["python3"]
CMD ["app.py"]