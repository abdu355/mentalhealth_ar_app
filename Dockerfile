FROM python:3.7-slim

RUN mkdir /streamlit

COPY requirements.txt /streamlit

WORKDIR /streamlit

# RUN apt-get -y update
# RUN apt-get -y install git

RUN pip install --upgrade pip
# RUN pip install --no-use-pep517 Google-Search-API
RUN pip install -r requirements.txt

COPY . /streamlit

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py"]