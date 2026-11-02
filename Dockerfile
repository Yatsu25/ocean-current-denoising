FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/app/src
CMD ["sh", "-c", "python -m ocean_denoise.train && python -m ocean_denoise.evaluate"]
