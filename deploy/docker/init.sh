#!/bin/bash

if [ -n "$RUN_JOB" ] ; then
    echo "Run job [$RUN_JOB]..."
    python /app/job.py $RUN_JOB
else
    SERVICES=()
    if [ -n "$APP_CRONTAB" ] && [ -s /etc/crontab ] && [ -n "$(grep '[^[:space:]]' /etc/crontab)" ] ; then
        SERVICES+=("crontab")
    fi
    if [ -n "$APP_API" ] ; then
        SERVICES+=("nginx" "gunicorn")
    fi
    if [ -n "$APP_QUEUE" ] ; then
        SERVICES+=("worker")
    fi

    if [ ${#SERVICES[@]} -gt 0 ] ; then
        echo "Run supervisord..."
        supervisord -c /etc/supervisord.conf

        for SERVICE in ${SERVICES[@]}; do
            echo "Start [$SERVICE]..."
            supervisorctl start $SERVICE
        done
    fi

    echo "App ready..."
    tail -f /dev/null
fi
