FROM python:3.10.2

RUN pip install -U pip

COPY requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt

# copy into a directory of its own (so it isn't in the toplevel dir)
COPY . /app
WORKDIR /app