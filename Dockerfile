FROM python:3.13-slim

WORKDIR /WGU_RUL_CAPSTONE

COPY /bearing_data /bearing_data
COPY /model_data /model_data
COPY /models /models
COPY /static /static
COPY /templates /templates
COPY / main.py
COPY / requirements.txt

RUN python3 -m venv venv
RUN source ./.venv/bin/activate
RUN pip install -r requirements.txt
RUN pip install gunicorn

EXPOSE 5000

CMD ['gunicorn', 'main:app', '-b', '0.0.0.0:5000', '-w', '4']