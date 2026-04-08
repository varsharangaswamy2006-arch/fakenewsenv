FROM python:3.10

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

EXPOSE 7860

CMD ["sh", "-c", "python inference.py & streamlit run app.py --server.port=7860 --server.address=0.0.0.0"]
