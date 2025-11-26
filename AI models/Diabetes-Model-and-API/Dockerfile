FROM python:3.10-slim
WORKDIR /
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8001
CMD ["python", "./start_server.py"]