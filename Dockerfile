FROM python:3.8-slim
WORKDIR /code

COPY pyproject.toml /code

# RUN apk update && apk add --update alpine-sdk

RUN pip install --upgrade pip && pip3 install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev --no-ansi --no-interaction

VOLUME ["/models"]

EXPOSE 8000
COPY ./app /code/app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]