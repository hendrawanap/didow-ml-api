FROM python:3.8.13-slim as build
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
build-essential gcc 

WORKDIR /usr/app
RUN python -m venv /usr/app/venv
ENV PATH="/usr/app/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install -r requirements.txt

FROM python:3.8.13-slim
RUN apt-get update
RUN apt-get install -y libgl1 libglib2.0-0
WORKDIR /usr/app/venv
COPY --from=build /usr/app/venv .
COPY . .

ENV PATH="/usr/app/venv/bin:$PATH"
CMD uvicorn main:app --host 0.0.0.0 --port $PORT