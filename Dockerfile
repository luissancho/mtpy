FROM python:3.11-slim

# Build dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        apt-utils \
        bash \
        build-essential \
        cmake \
        curl \
        dnsutils \
        gcc \
        libpq-dev \
        nginx \
        supervisor \
        vim \
        wget \
        zip \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSLO "https://github.com/aptible/supercronic/releases/download/v0.2.29/supercronic-linux-amd64" \
    && echo "cd48d45c4b10f3f0bfdd3a57d054cd05ac96812b supercronic-linux-amd64" | sha1sum -c - \
    && chmod +x supercronic-linux-amd64 \
    && mv supercronic-linux-amd64 /usr/local/bin/supercronic-linux-amd64 \
    && ln -s /usr/local/bin/supercronic-linux-amd64 /usr/local/bin/supercronic \
    && ln -sf /bin/bash /bin/sh

# Setup application root
RUN mkdir -p /app
WORKDIR /app

# Python requirements
COPY ./requirements.txt ./
RUN pip install -r ./requirements.txt

# Configure crontab schedules
COPY ./deploy/docker/crontab /etc/crontab

# Configure NGINX
COPY ./deploy/docker/nginx.conf /etc/nginx/nginx.conf

# Configure Supervisord
COPY ./deploy/docker/supervisord.conf /etc/supervisord.conf

# Init script
COPY ./deploy/docker/init.sh /usr/local/bin/init.sh
RUN chmod +x /usr/local/bin/init.sh

# App source code
COPY . .
RUN mkdir -p files log

# Run container as non-root user and set permissions
RUN addgroup --gid 10001 docker \
    && adduser --disabled-password --uid 10001 --ingroup docker docker \
    && chown -R docker:docker /app /run /var/log /var/cache /var/lib

USER docker
EXPOSE 8042

# Init script
CMD ["/bin/bash", "-c", "init.sh"]
