FROM nginx:alpine

RUN apk add --no-cache openssl

# nginx.conf is a template: ${FIREWALL_UPSTREAM} is substituted at start-up
# (default mitmproxy:8080; the colo override uses 127.0.0.1:8080 with host networking).
COPY nginx.conf /etc/nginx/nginx.conf.template
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 443

ENTRYPOINT ["/entrypoint.sh"]
