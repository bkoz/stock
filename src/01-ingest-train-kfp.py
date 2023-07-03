import kfp
import kfp.dsl as dsl
import kfp.components as comp
from kfp.components import func_to_container_op, InputPath, OutputPath
from kubernetes.client.models import V1EnvVar
import os

#
# Read stock market data from Yahoo and save to a csv file.
#
# @func_to_container_op
def ingest(ticker: str, out_value: OutputPath(str)) -> str:
    
    from pandas_datareader import data as pdr
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import yfinance as yf
    import os
    from dotenv import load_dotenv
    from minio import Minio
    from minio.error import S3Error

    def push_data(bucket: str, out_value: str):
        # Create a client with the MinIO server playground, its access key
        # and secret key.
        client = Minio(
            os.getenv("S3_ENDPOINT"), os.getenv("ACCESS_KEY"), os.getenv("SECRET_KEY")
        )

        # Create a bucket if it does not exist.
        found = client.bucket_exists(bucket)
        if not found:
            client.make_bucket(bucket)
        else:
            print(f"Bucket {bucket} already exists")

        #
        # Upload 'out_value' as object name
        # 'out_value' to the bucket 'bucket'.
        #
        try:
            client.fput_object(bucket, out_value, out_value)

        except S3Error as err:
            print(err)


    load_dotenv(override=True)
    yf.pdr_override()
    df = pdr.get_data_yahoo(ticker, start="2023-01-01", end="2023-06-01")
    print(f'******* Env S3_ENDPOINT = {os.getenv("S3_ENDPOINT")}')
    print(f'******* Ticker = {ticker}')
    print(f"Saving {out_value} to local storage.")
    df.to_csv(out_value)
    fd = open(out_value,'r')
    d = fd.read()
    fd.close()
    print(f"Pushing {out_value} to object store.")
    push_data('data', out_value)
    #
    # If I print the file contents, they get stored as an artifact.
    #
    # print(d)
    print("Finished ingest()")
    return "Finished ingest()"

#
# Create the data ingest component.
#
ingest_op = comp.create_component_from_func(ingest, base_image='quay.io/bkozdemb/pipestage:latest')


#
# Train function
#
# @func_to_container_op
def train(csv_file: kfp.components.InputPath(), plotfile: kfp.components.OutputPath(str)) -> str:
    import pandas as pd
    print(f'CSV file = {csv_file}')
    df = pd.read_csv(csv_file)
    print(df.head())
    #
    # Predict stock prices with Long short-term memory (LSTM)
    # This simple example will show you how LSTM models predict time series data.
    # Stock market data is a great choice for this because it's quite regular and widely available via the Internet.
    #
    # Introduction
    # LSTMs are very powerful in sequence prediction problems. They can store past information.
    # Loading the dataset
    # Use the pandas-data reader to get the historical stock prices from Yahoo! finance.
    # For this example, I get only the historical data till the end of *training_end_data*.
    #

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pandas_datareader import data as pdr
    import yfinance as yf
    import tensorflow as tf
    import os
    from minio import Minio
    from minio.error import S3Error
    import tf2onnx
    import onnx
    import glob
    from dotenv import load_dotenv

    def push_file_to_s3(bucket: str, filename: str) -> None:
        # Create a client with the MinIO server playground, its access key
        # and secret key.
        client = Minio(
            os.getenv("S3_ENDPOINT"), os.getenv("ACCESS_KEY"), os.getenv("SECRET_KEY")
        )

        # Create a 'models' bucket if it does not exist.
        found = client.bucket_exists(bucket)
        if not found:
            client.make_bucket(bucket)
        else:
            print("Bucket already exists")

        # Upload 'stocks.onnx' as object name
        # 'stock-model.onnx' to the bucket 'models'.
        #
        # fput_object(<bucket>, <destination-out_value>, <source-out_value>)
        #
        try:
            client.fput_object(bucket, filename, filename)

        except S3Error as err:
            print(err)


    def push_model():
        # Create a client with the MinIO server playground, its access key
        # and secret key.
        client = Minio(
            os.getenv("S3_ENDPOINT"), os.getenv("ACCESS_KEY"), os.getenv("SECRET_KEY")
        )

        # Create a 'models' bucket if it does not exist.
        found = client.bucket_exists("models")
        if not found:
            client.make_bucket("models")
        else:
            print("Bucket 'models' already exists")

        # Upload 'stocks.onnx' as object name
        # 'stock-model.onnx' to the bucket 'models'.
        #
        # fput_object(<bucket>, <destination-out_value>, <source-out_value>)
        #
        try:
            client.fput_object("models", "stocks.onnx", "stocks.onnx")

        except S3Error as err:
            print(err)


    def upload_local_directory_to_minio(local_path, bucket_name, minio_path):
        # Create a client with the MinIO server playground, its access key
        # and secret key.
        client = Minio(
            os.getenv("S3_ENDPOINT"), os.getenv("ACCESS_KEY"), os.getenv("SECRET_KEY")
        )

        assert os.path.isdir(local_path)

        for local_file in glob.glob(local_path + "/**"):
            local_file = local_file.replace(os.sep, "/")  # Replace \ with / on Windows
            if not os.path.isfile(local_file):
                upload_local_directory_to_minio(
                    local_file, bucket_name, minio_path + "/" + os.path.basename(local_file)
                )
            else:
                remote_path = os.path.join(minio_path, local_file[1 + len(local_path) :])
                remote_path = remote_path.replace(
                    os.sep, "/"
                )  # Replace \ with / on Windows
                client.fput_object(bucket_name, remote_path, local_file)


    load_dotenv(override=True)
    print(f'******* Env ACCESS_KEY = {os.getenv("ACCESS_KEY")}')
    print(f'******* Env SECRET_KEY = {os.getenv("SECRET_KEY")}')
    print(f'******* Env S3_ENDPOINT = {os.getenv("S3_ENDPOINT")}')

    tickers = "IBM"
    start_date = "1980-12-01"
    end_date = "2018-12-31"

    # yf.pdr_override()
    # stock_data = pdr.get_data_yahoo(tickers, start_date)
    print(f'Reading data from {csv_file}')
    stock_data = pd.read_csv(csv_file)
    stock_data_len = stock_data["Close"].count()
    print(f"Read in {stock_data_len} stock values")


    close_prices = stock_data.iloc[:, 1:2].values
    # print(close_prices)

    # Some of the weekdays might be public holidays in which case no price will be available.
    # For this reason, we will fill the missing prices with the latest available prices

    all_bussinessdays = pd.date_range(start=start_date, end=end_date, freq="B")
    print(all_bussinessdays)

    close_prices = stock_data.reindex(all_bussinessdays)
    close_prices = stock_data.fillna(method="ffill")

    # The dataset is now complete and free of missing values. Let's have a look to the data frame summary:
    # Feature scaling

    training_set = close_prices.iloc[:, 1:2].values

    from sklearn.preprocessing import MinMaxScaler

    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    # print(training_set_scaled.shape)

    # LSTMs expect the data in a specific format, usually a 3D tensor. I start by creating data with 60 days and converting it into an array using NumPy.
    # Next, I convert the data into a 3D dimension array with feature_set samples, 60 days and one feature at each step.
    features = []
    labels = []
    for i in range(60, stock_data_len):
        features.append(training_set_scaled[i - 60 : i, 0])
        labels.append(training_set_scaled[i, 0])

    features = np.array(features)
    labels = np.array(labels)

    features = np.reshape(features, (features.shape[0], features.shape[1], 1))

    #
    # Feature tensor with three dimension: features[0] contains the ..., features[1] contains the last 60 days of values and features [2] contains the  ...
    #
    # Create the LSTM network
    # Let's create a sequenced LSTM network with 50 units. Also the net includes some dropout layers with 0.2 which means that 20% of the neurons will be dropped.

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.LSTM(
                units=50, return_sequences=True, input_shape=(features.shape[1], 1)
            ),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(units=50, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(units=50, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(units=50),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(units=1),
        ]
    )

    print(model.summary())

    # The model will be compiled and optimize by the adam optimizer and set the loss function as mean_squarred_error

    model.compile(optimizer="adam", loss="mean_squared_error")

    #
    # For testing purposes, train for 2 epochs. This should be increased to improve model accuracy.
    #
    from time import time

    start = time()
    history = model.fit(features, labels, epochs=2, batch_size=32, verbose=1)
    end = time()

    print("Total training time {} seconds".format(end - start))

    #
    # Save the model
    #
    print("Saving the model locally...")

    #
    # Tensorflow "saved_model" format.
    #
    tf.keras.models.save_model(model, "stocks/1")

    #
    # onnx format
    #
    # input_signature = [tf.TensorSpec([3, 3], tf.float32, name='x')]
    # onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
    model_proto, _ = tf2onnx.convert.from_keras(model)
    onnx.save(model_proto, "stocks.onnx")

    #
    # For now, get validation data from Yahoo.
    #
    yf.pdr_override()
    testing_start_date = "2019-01-01"
    testing_end_date = "2019-04-10"

    test_stock_data = pdr.get_data_yahoo(tickers, testing_start_date, testing_end_date)

    test_stock_data_processed = test_stock_data.iloc[:, 1:2].values

    all_stock_data = pd.concat((stock_data["Close"], test_stock_data["Close"]), axis=0)

    inputs = all_stock_data[len(all_stock_data) - len(test_stock_data) - 60 :].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)

    X_test = []
    for i in range(60, 129):
        X_test.append(inputs[i - 60 : i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    print("Pushing model to object storage...")
    print(f'S3_ENDPOINT = {os.getenv("S3_ENDPOINT")},ACCESS_KEY = {os.getenv("ACCESS_KEY")}, SECRET_KEY = {os.getenv("SECRET_KEY")}')
    push_model()
    upload_local_directory_to_minio("stocks", "models", "stocks")

    #
    # Plots
    #
    plt.figure(figsize=(10,6))
    plt.plot(test_stock_data_processed, color='blue', label=f'Actual Stock Price')
    plt.plot(predicted_stock_price , color='red', label=f'Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel(f'Stock Price')
    plt.legend()
    plt.savefig('plot.png')
    plt.show()
    push_file_to_s3('plots', 'plot.png')
    return "train() finished"

#
# Create the data ingest component.
#
train_op = comp.create_component_from_func(train, base_image='quay.io/bkozdemb/pipestage:latest')

#
# Validate
#
# @func_to_container_op
def validate(plotfile: kfp.components.InputPath()) -> str:

    from minio import Minio
    from minio.error import S3Error
    import os
    from dotenv import load_dotenv
    
    def push_s3(bucket: str, out_value: str):
        load_dotenv(override=True)

        # Create a client with the MinIO server playground, its access key
        # and secret key.
        client = Minio(
            os.getenv("S3_ENDPOINT"), os.getenv("ACCESS_KEY"), os.getenv("SECRET_KEY")
        )

        # Create a bucket if it does not exist.
        found = client.bucket_exists(bucket)
        if not found:
            client.make_bucket(bucket)
        else:
            print(f"Bucket {bucket} already exists")

        #
        # Upload 'out_value' as object name
        # 'out_value' to the bucket 'bucket'.
        #
        try:
            client.fput_object(bucket, out_value, out_value)

        except S3Error as err:
            print(err)

    print(f'Saving plotfile as {plotfile}.')
    push_s3('plots', plotfile)
    return "Finished validate()"

validate_op = comp.create_component_from_func(validate, base_image='quay.io/bkozdemb/pipestage:latest')

# Advanced function
# Demonstrates imports, helper functions and multiple outputs

from typing import NamedTuple

@dsl.pipeline(
   name='ingest-train-pipeline',
   description='A simple pipeline that downloads stock data and saves it.'
)
#
# Currently kfp-tekton doesn't support paremeter passing to the pipelinerun yet, 
# so we hard code the ticker string here.
#
def ingest_train_pipeline(ticker = 'IBM'):
    # Passing pipeline parameter and a constant value as operation arguments
    # add_task = add_op(a, 4) # Returns a dsl.ContainerOp class instance.
    from dotenv import load_dotenv

    load_dotenv(override=True)
    ingest_task = ingest_op(ticker)\
      .add_env_variable(V1EnvVar(name='S3_ENDPOINT', value=os.getenv('S3_ENDPOINT')))\
      .add_env_variable(V1EnvVar(name='ACCESS_KEY', value=os.getenv('ACCESS_KEY')))\
      .add_env_variable(V1EnvVar(name='SECRET_KEY', value=os.getenv('SECRET_KEY')))

    train_task = train_op(ingest_task.outputs['out_value'])\
      .add_env_variable(V1EnvVar(name='S3_ENDPOINT', value=os.getenv('S3_ENDPOINT')))\
      .add_env_variable(V1EnvVar(name='ACCESS_KEY', value=os.getenv('ACCESS_KEY')))\
      .add_env_variable(V1EnvVar(name='SECRET_KEY', value=os.getenv('SECRET_KEY')))
    
    # validate_task = validate_op(train_task.outputs['plotfile'])\
    #   .add_env_variable(V1EnvVar(name='S3_ENDPOINT', value=os.getenv('S3_ENDPOINT')))\
    #   .add_env_variable(V1EnvVar(name='ACCESS_KEY', value=os.getenv('ACCESS_KEY')))\
    #   .add_env_variable(V1EnvVar(name='SECRET_KEY', value=os.getenv('SECRET_KEY')))


if __name__ == '__main__':

    import kfp_tekton
    import urllib
    from dotenv import load_dotenv

    load_dotenv(override=True)

    # from kfp_tekton.compiler import TektonCompiler
    # TektonCompiler().compile(ingest_train_pipeline, __file__.replace('.py', '.yaml'))

    # kfp                      1.8.19
    # kfp-pipeline-spec        0.1.16
    # kfp-server-api           1.8.5
    # kfp-tekton               1.5.2
    # kfp-tekton-server-api    1.5.0

    kubeflow_endpoint = os.environ["KUBEFLOW_ENDPOINT"]
    bearer_token = os.environ["BEARER_TOKEN"]

    print(f'kubeflow_endpoint = {kubeflow_endpoint}')
    print(f'bearer_token = {bearer_token}')
    client = kfp_tekton.TektonClient(
        host=urllib.parse.urljoin(kubeflow_endpoint, "/"),
        existing_token=bearer_token,
    )
    # print(client.list_experiments())

    client.create_run_from_pipeline_func(
        ingest_train_pipeline, arguments=None, experiment_name="ingest-train-pipeline")
