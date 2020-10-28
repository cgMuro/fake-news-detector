FROM python:3.8-slim
COPY ./use_svm.py /deploy/
COPY ./requirements.txt /deploy/
COPY ./final_svm.pkl /deploy/
COPY ./glove.6B.100d.txt /deploy/
COPY ./nltk.txt /deploy/
WORKDIR /deploy/
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader punkt
ENTRYPOINT ["python", "use_svm.py"]