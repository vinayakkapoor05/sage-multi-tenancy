FROM waggle/plugin-base:1.1.1-ml

COPY . /app/
WORKDIR /app

RUN pip3 install --no-cache-dir -e .
RUN chmod +x /app/plugin-entrypoint.sh

ENTRYPOINT ["/app/plugin-entrypoint.sh"]
