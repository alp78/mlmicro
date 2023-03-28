import json
from os import listdir, getpid
from os.path import isfile, join
from enum import Enum
import asyncio
import sys
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from fastapi import FastAPI, status, Response
from fastapi import Form
import aioredis
import collections
from numpy import ndarray
import numpy as np
import joblib
from pydantic import BaseModel
from pydantic import Field
from os import path
from typing import List, Tuple, Optional
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import time
from os import remove

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# function to flatten a nested dictionary with nested lists
def flatten(d, sep="_"):

    obj = collections.OrderedDict()

    def recurse(t,parent_key=""):
        
        if isinstance(t,list):
            for i in range(len(t)):
                recurse(t[i],parent_key + sep + str(i) if parent_key else str(i))
        elif isinstance(t,dict):
            for k,v in t.items():
                recurse(v,parent_key + sep + k if parent_key else k)
        else:
            obj[parent_key] = t

    recurse(d)

    return obj


# helper function for /train
def fit_model(model_name, model, train_data, train_targets):
    model.fit(train_data, train_targets)
    job_file = f'/models/digits_{model_name}_model.joblib'
    job_targets = (model, train_targets)
    joblib.dump(job_targets, job_file)
    
    return {'model': path.abspath(job_file)}

# helper function for /predict 
async def load_input_data():
    test_data_bytes = await redis.get('test_data')
    return PredIn(data = np.fromstring(test_data_bytes).reshape(450, 64))

class InputType(str, Enum):
    DCVS = 'dcvs'
    EWF = 'ewf'
    ICS2 = 'ics2'
    

class InputModel(str, Enum):
    GAUSS = 'gauss'
    SVC = 'svc'

# prediction model input: numpy array
class PredIn(BaseModel):
    data: ndarray = Field(...)
    class Config:
        arbitrary_types_allowed = True

# model output: list of integers representing categories for each list of the input array
class PredOut(BaseModel):
    category: List[int] 

# Model class to represent each model (svc, gauss) with methods to load and predict
class Model:
    model: Optional[GaussianNB or GridSearchCV]
    targets: Optional[ndarray]
    input_model: Optional[InputModel]
    
    def __init__(self, input_model: InputModel):
        self.input_model = input_model
    
    def load_model(self):
        model_file = f'/models/digits_{self.input_model}_model.joblib'
        model_tuple: Tuple[GaussianNB or GridSearchCV, ndarray] = joblib.load(model_file)
        model, targets = model_tuple
        self.model = model
        self.targets = targets
    
    def predict(self, input_data: PredIn, start: float):
        # print(('predict', threading.current_thread().name, multiprocessing.current_process().name))
        # sys.stdout.flush()

        prediction = self.model.predict(input_data.data)
        res = PredOut(category=prediction.tolist()[:5])
        stop = time.time()
        total = stop - start
        print(total)
        print(' ')

        with open('/models/times.txt', 'a') as f:
            f.write(str(f'{total}\n'))        
        
        
        return res

app = FastAPI()
redis = aioredis.from_url('redis://redis', encoding='utf-8', decode_responses=False)

model_digits_gauss = Model('gauss')
model_digits_svc = Model('svc')

executor = ThreadPoolExecutor(max_workers=8)
# executor = ProcessPoolExecutor(max_workers=8)

print(('main', threading.current_thread().name))


# http -v GET http://localhost:8000/load_model
@app.get('/list_model')
async def list_model():

    model_dir = '/models'
    models = [f for f in listdir(model_dir) if isfile(join(model_dir, f))]
    return models

# simply returns the JSON input as nested dictionary
# http -v GET "http://localhost:8000/read_json?input_type=dcvs"
@app.get('/read_json')
async def read_json(input_type: InputType):
    filepath = f'/code/app/input/request_{input_type}.json'

    with open(filepath) as f:
        cnt = json.load(f)
        
    return cnt

# retrieve JSON stored in Redis and returns flatten dic
# http -v GET "http://127.0.0.1:8000/read_redis?r_key=[dcvs|ewf|ics2]"
@app.get('/read_redis')
async def read_redis(r_key: InputType,
                     response: Response):
    # async with redis.client() as r:
    
    if await redis.exists(r_key) == 1:
        r_cnt = await redis.get(r_key)
        jload = json.loads(r_cnt)
        vals = flatten(jload)
    else:
        response.status_code = status.HTTP_400_BAD_REQUEST
        r_keys = await redis.keys()
        res = {'error': f'key [{r_key}] does not exist', 'keys': r_keys}
        return res
    
    return vals


# delete all keys in Redis
# http -v GET http://127.0.0.1:8000/del_keys
@app.get('/del_keys')
async def del_keys():
    r_keys = await redis.keys()
    if len(r_keys) == 0:
        return 'no keys found'
    else:
        for key in await redis.keys():
            await redis.delete(key)
    
        return {'deleted keys': r_keys}


# extract test data (ndarray) from Digits dataset and store it as bytes string in Redis
# http -v GET http://127.0.0.1:8000/store_test
@app.get('/store_test')
async def store_test():
    digits = load_digits()
    data = digits.data
    targets = digits.target
    _, test_data, _, test_targets = train_test_split(data, targets, random_state=0)
    
    # saving test data for further re-training
    test_data_bytes = test_data.tostring()
    await redis.set('test_data', test_data_bytes)
    
    # saving testr targets to compute model accuracy 
    test_targets_bytes = test_targets.tostring()
    await redis.set('test_targets', test_targets_bytes)
    
    
    return {'redis_key': 'test_data', 
            'shape_data': test_data.shape, 
            'shape_targets': test_targets.shape}


# write input JSON to redis as JSON (first converted to dic)
# http -v --form POST http://127.0.0.1:8000/write_redis input_type=[dcvs|ewf|ics2]
@app.post('/write_redis')
async def write_redis(input_type: InputType = Form(...)):
    root_dir = '/code'
    filepath = path.join(root_dir, 'app', 'input', f'request_{input_type}.json') 

    with open(filepath) as f:
        cnt = json.load(f)

    jdump = json.dumps(cnt)

    r_key = cnt['ShpPcdDetailsMsg']['Hdr']['Sndr']['@AppCd'].strip().lower()
        
    await redis.set(r_key, jdump)
    return {'redis key': r_key}


# apply prediction    
# http -v --form POST http://127.0.0.1:8000/predict input_model=svc
@app.post('/predict')
async def predict(input_type: InputType = Form(...),
                  input_model: InputModel = Form(...)):
    start = time.time()
    loop = asyncio.get_event_loop()
    
    root_dir = '/code'
    filepath = join(root_dir, 'app', 'input', f'request_{input_type}.json')
    with open(filepath) as f:
        cnt = json.load(f)
    
    r_key = cnt['ShpPcdDetailsMsg']['Hdr']['Sndr']['@AppCd'].strip().lower()
    json_str = json.dumps(cnt)
    await redis.set(r_key, json_str)

    # print(getpid())
    # print(threading.current_thread().name)
    # print(r_key)

    r_cnt = await redis.get(r_key)
    jload = json.loads(r_cnt)
    vals = flatten(jload)

    
    if input_model == InputModel.SVC:
        test_data_pred = await load_input_data()
        res = await loop.run_in_executor(executor, model_digits_svc.predict, test_data_pred, start)
        return res

    if input_model == InputModel.GAUSS:
        test_data_pred = await load_input_data()
        res = await loop.run_in_executor(executor, model_digits_gauss.predict, test_data_pred, start)
        return res

    # if input_model == InputModel.SVC:
    #     output = await model_digits_svc.predict()
    #     return output
    
    # if input_model == InputModel.GAUSS:
    #     output = await model_digits_gauss.predict()
    #     return output


# train models, dumping the trained joblib models in local folder
# http -v --form POST http://127.0.0.1:8000/train input_model=[svc|gauss]
@app.post('/train')
async def train(input_model: InputModel = Form(...)):
  
    digits = load_digits()
    data = digits.data
    targets = digits.target
    train_data, _, train_targets, _ = train_test_split(data, targets, random_state=0)
    root_dir = '/code'
    
    if input_model == InputModel.SVC:
        param_grid = {
            'C': [1, 1, 100, 1000],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
        }
        model_svc = GridSearchCV(SVC(), param_grid)
        return fit_model(input_model, model_svc, train_data, train_targets)

    if input_model == InputModel.GAUSS:
        model_gauss = GaussianNB()
        return fit_model(input_model, model_gauss, train_data, train_targets)
        


# code executed at application startup (uvicorn start/reload)
@app.on_event('startup')
async def startup():
    app.state.executor = ProcessPoolExecutor()
    # check if test_data is in Redis, if not, create it
    if not await redis.exists('test_data'):
        digits = load_digits()
        data = digits.data
        targets = digits.target
        _, test_data, _, _ = train_test_split(data, targets, random_state=0)
        
        test_data_bytes = test_data.tostring()
        await redis.set('test_data', test_data_bytes)
    
    model_digits_gauss.load_model()
    model_digits_svc.load_model()


@app.on_event('shutdown')
async def shutdown():
    
    with open('/models/times.txt', 'r') as f:
        l = []
        [l.append(float(t)) for t in f.readlines()]
        avg = sum(l) / len(l)

    with open('/models/avg_furs.txt', 'w') as f:
        f.write(str(avg * 1000))
    
    remove('/models/times.txt')
    app.state.executor.shutdown()
