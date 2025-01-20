#!/bin/bash

python manage.py makemigrations
python manage.py migrate
# gunicorn quiz_api.wsgi:application --bind 0.0.0.0:$PORT
python manage.py runserver 0.0.0.0:$PORT
