FROM python:3.6

RUN mkdir /code
WORKDIR /code
ADD . /code/
RUN pip install pip flask numpy scipy sklearn scikit_learn statsmodels pandas
RUN pip install mord

EXPOSE 5000
CMD ["python", "/code/PredictorService.py"]