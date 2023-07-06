FROM python:3.10
# set work directory
WORKDIR /srv/magicbooth/training_bot
# set telegram bot token
ENV TOKEN="TOKEN"
# copy project
COPY . /srv/magicbooth/training_bot

# install dependencies
RUN pip install -r ./requirements.txt

CMD ["python", "main.py"]
