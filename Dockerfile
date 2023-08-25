FROM python:3.10.2

WORKDIR /chatbot
ADD ./chatbot/requirements.txt ./chatbot/requirements.txt
RUN pip install -U pip && pip install -r ./chatbot/requirements.txt

COPY ./chatbot ./chatbot
WORKDIR ./chatbot