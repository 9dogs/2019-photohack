FROM python:3.6.8

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "gunicorn", "-w 8", "-b 0.0.0.0:80", "photohack.app:app" ]
