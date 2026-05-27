#!/bin/sh
set -e

CERT_DIR=/etc/nginx/certs
mkdir -p "$CERT_DIR"

if [ ! -f "$CERT_DIR/nginx.crt" ]; then
    echo "[nginx-entrypoint] Generating self-signed SSL certificate..."
    openssl req -x509 -nodes -days 3650 -newkey rsa:2048 \
        -keyout "$CERT_DIR/nginx.key" \
        -out    "$CERT_DIR/nginx.crt" \
        -subj "/C=US/ST=Dev/L=Local/O=LLMProxy/CN=nginx" \
        -addext "subjectAltName=DNS:nginx,DNS:localhost,IP:127.0.0.1"
    echo "[nginx-entrypoint] Certificate written to $CERT_DIR/nginx.crt"
fi

exec nginx -g "daemon off;"
