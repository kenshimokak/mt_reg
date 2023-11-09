from sklearn.linear_model import ARDRegression
from sklearn.pipeline import make_pipeline
from skl2onnx import convert_sklearn
from onnx import onnx
import onnxruntime as rt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import matplotlib.pyplot as plt
from datetime import datetime
import MetaTrader5 as mt5
import pytz

# Initialize MetaTrader 5
init = mt5.initialize(path="D:\\Program Files\\MetaTrader 5\\terminal64.exe")
if not init:
    print("initialize() failed, error code =", mt5.last_error())

# Log in to your MetaTrader 5 account
authorized = mt5.login(68936836, "5myxghhb", "MetaQuotes-Demo")
if authorized:
    print("Connected to account: ......... # \n\n {}".format(mt5.account_info()._asdict()))

# Define global variables
SYMBOL = 'XAUUSD'
TIMEFRAME = mt5.TIMEFRAME_M30
MA_PERIODS = [i for i in range(30, 1000, 30)]
MARKUP = 0.0006
TSTART_DATE = datetime(2019, 1, 1)
START_DATE = datetime(2021, 1, 1)
STOP_DATE = datetime(2023, datetime.now().month, timedelta(weeks=2).days)
BAD_SAMPLES_BOOK = pd.DatetimeIndex([])
ITERATION = 1000
STOP_ROUNDED = 30
CAT_DEPTH = 6
MAX_RES = 0.999
LOOP_RANG = 10

# Establish connection to the MetaTrader 5 terminal
def mt5_connection():
    init = mt5.initialize(path="D:\\Program Files\\MetaTrader 5\\terminal64.exe")
    if not init:
        print("initialize() failed, error code =", mt5.last_error())
        return False
    return True

# Log in to a MetaTrader 5 account
def login_mt5(account, password=None, server=None):
    authorized = mt5.login(account) if password is None else mt5.login(account, password, server)
    if authorized:
        print("Connected to account: ......... #{}".format(account))
        print(mt5.account_info())
    else:
        print("failed to connect at account #{}, error code: {}".format(account, mt5.last_error()))
    return authorized

# Apply a function for all MetaTrader 5 accounts
def apply_for_all_accounts(func, *args, **kwargs):
    for key, user in users.items():
        login_mt5(user['account'], user['password'], user['server'])
        func(*args, **kwargs)
        print(type(args))
        
# Get historical price data from MetaTrader 5
def get_prices() -> pd.DataFrame:
    # Assuming you have already established a connection with MT5
    mt5_data = mt5.copy_rates_range(SYMBOL, TIMEFRAME, TSTART_DATE, STOP_DATE)
    pFixed = pd.DataFrame(mt5_data, columns=['time', 'close']).set_index('time')
    pFixed.index = pd.to_datetime(pFixed.index, unit='s')

    pFixed = pFixed.dropna()
    pFixedC = pFixed.copy()

    count = 0
    for i in MA_PERIODS:
        pFixed[str(count)] = pFixedC['close'] - pFixedC['close'].rolling(i).mean()
        count += 1

    return pFixed.dropna()

def labelling_relabeling(dataset, min=15, max=35, relabeling=False):
    labels = []
    for i in range(dataset.shape[0]-max):
        rand = random.randint(min, max)
        curr_pr = dataset['close'][i]
        future_pr = dataset['close'][i + rand]

        if relabeling:
            m_labels = dataset['meta_labels'][i:rand+1].values

        if relabeling and 0.0 in m_labels:
            labels.append(2.0)
        else:
            if future_pr + MARKUP < curr_pr:
                labels.append(1.0)
            elif future_pr - MARKUP > curr_pr:
                labels.append(0.0)
            else:
                labels.append(2.0)

    dataset = dataset.iloc[:len(labels)].copy()
    dataset['labels'] = labels
    dataset = dataset.dropna()
    dataset = dataset.drop(
        dataset[dataset.labels == 2].index)

    return dataset

# Tester function for evaluating the trading strategy
def tester(dataset: pd.DataFrame, markup=0.0, use_meta=False, plot=False):
    last_deal = int(2)
    last_price = 0.0
    report = [0.0]
    meta_labels = dataset['labels'].copy()

    for i in range(dataset.shape[0]):
        pred = dataset['labels'][i]
        meta_labels[i] = np.nan
        if use_meta:
            pred_meta = dataset['meta_labels'][i]  # 1 = allow trades

        if last_deal == 2 and ((use_meta and pred_meta == 1) or not use_meta):
            last_price = dataset['close'][i]
            last_deal = 0 if pred <= 0.5 else 1
            continue

        if last_deal == 0 and pred > 0.5 and ((use_meta and pred_meta == 1) or not use_meta):
            last_deal = 2
            report.append(report[-1] - markup +
                          (dataset['close'][i] - last_price))
            if report[-1] > report[-2]:
                meta_labels[i] = 1
            else:
                meta_labels[i] = 0
            continue

        if last_deal == 1 and pred < 0.5 and ((use_meta and pred_meta == 1) or not use_meta):
            last_deal = 2
            report.append(report[-1] - markup +
                          (last_price - dataset['close'][i]))
            if report[-1] > report[-2]:
                meta_labels[i] = 1
            else:
                meta_labels[i] = 0

    y = np.array(report).reshape(-1, 1)
    X = np.arange(len(report)).reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(X, y)

    l = lr.coef_
    if l >= 0:
        l = 1
    else:
        l = -1

    if (plot):
        plt.plot(report)
        plt.plot(lr.predict(X))
        plt.title("Strategy performance R^2 " + str(format(lr.score(X, y) * l, ".2f")))
        plt.xlabel("the number of trades")
        plt.ylabel("cumulative profit in pips")
        plt.show()

    return lr.score(X, y) * l, meta_labels.fillna(method='backfill')



# ...

# Define the ARDRegression model and a pipeline
model = ARDRegression()
pipeline = make_pipeline(model)

# Learn the ARDRegression model with train and validation subsets
pipeline.fit(train_X, train_y)

# Convert the ARDRegression model to ONNX
onnx_model = convert_sklearn(pipeline, initial_types=[('input', FloatTensorType([None, train_X.shape[1]]))])

# Save the ONNX model to a file
with open('ard_regression_model.onnx', "wb") as f:
    f.write(onnx_model.SerializeToString())

# Load the ONNX model using ONNX Runtime
onnx_session = rt.InferenceSession('ard_regression_model.onnx')

# Predict using the ONNX model
def predict_with_onnx(features):
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    input_data = {input_name: np.array([features], dtype=np.float32)}
    output = onnx_session.run([output_name], input_data)
    return output[0]

# Replace the CatBoost model prediction with ARDRegression model prediction
def catboost_model_1(features):
    return predict_with_onnx(features)

# Export the ARDRegression model to MQL code
export_model_to_MQL_code([res[-1][0], model, pr], str(1))
