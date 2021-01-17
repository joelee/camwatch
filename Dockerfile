FROM python:3.9-buster

RUN mkdir /config
RUN mkdir /recordings
RUN mkdir -p /opt/camwatch/src

RUN apt-get update
RUN apt-get install -y --no-install-recommends git

COPY cam_detector /opt/camwatch/src/
COPY config/camwatch-quick_start.yaml /config/camwatch.yaml
COPY config/camwatch-defaults.yaml /config/camwatch-defaults.yaml

COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt
RUN rm /tmp/requirements.txt

RUN groupadd -g 1001 camwatch
RUN useradd -u 1001 -g camwatch -m -s /bin/bash camwatch
RUN chown -R camwatch:camwatch /recordings

ENV CV_CONFIG_PATH /config

USER camwatch
WORKDIR /opt/camwatch

CMD ["bash"]
# CMD ["python", "./main.py"]

