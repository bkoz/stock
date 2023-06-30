import kfp
import kfp.dsl as dsl
import kfp.components as comp
from kubernetes.client.models import V1EnvVar
import os

#
# Read stock market data from Yahoo and save to a csv file.
#
def ingest(ticker: str, filename: kfp.components.OutputPath(str)) -> str:
    
    from pandas_datareader import data as pdr
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import yfinance as yf
    import os
    from dotenv import load_dotenv
    from minio import Minio
    from minio.error import S3Error

    def push_data(bucket: str, filename: str):
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
        # Upload 'filename' as object name
        # 'filename' to the bucket 'bucket'.
        #
        try:
            client.fput_object(bucket, filename, filename)

        except S3Error as err:
            print(err)


    load_dotenv(override=True)
    # ticker = "IBM"
    # filename = ticker + ".csv"
    yf.pdr_override()
    df = pdr.get_data_yahoo(ticker, start="2023-01-01", end="2023-06-01")
    # print(f'******* Env ACCESS_KEY = {os.getenv("ACCESS_KEY")}')
    # print(f'******* Env SECRET_KEY = {os.getenv("SECRET_KEY")}')
    print(f'******* Env S3_ENDPOINT = {os.getenv("S3_ENDPOINT")}')
    print(f'******* Ticker = {ticker}')
    print(f"Saving {filename} to local storage.")
    df.to_csv(filename)
    fd = open(filename,'r')
    d = fd.read()
    fd.close()
    print(f"Pushing {filename} to object store.")
    push_data('data', filename)
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
def train(csv_file: kfp.components.InputPath()):
    import pandas as pd
    print(f'CSV file = {csv_file}')
    df = pd.read_csv(csv_file)
    print(df.head())
    return True

#
# Create the data ingest component.
#
train_op = comp.create_component_from_func(train, base_image='quay.io/bkozdemb/pipestage:latest')

# Define a Python function
def add(a: float, b: float) -> float:
   """Calculates sum of two arguments"""
   return a + b

add_op = comp.create_component_from_func(add, base_image='quay.io/hukhan/python:alpine3.6')

# Advanced function
# Demonstrates imports, helper functions and multiple outputs

from typing import NamedTuple


def my_divmod(dividend: float, divisor:float) -> NamedTuple('MyDivmodOutput', [('quotient', float), ('remainder', float), ('mlpipeline_ui_metadata', 'UI_metadata'), ('mlpipeline_metrics', 'Metrics')]):
    """Divides two numbers and calculate the quotient and remainder"""

    # Imports inside a component function:
    import numpy as np

    # This function demonstrates how to use nested functions inside a component function:
    def divmod_helper(dividend, divisor):
        return np.divmod(dividend, divisor)

    (quotient, remainder) = divmod_helper(dividend, divisor)

    from tensorflow.python.lib.io import file_io
    import json

    # Exports a sample tensorboard:
    metadata = {
      'outputs' : [{
        'type': 'tensorboard',
        'source': 'gs://ml-pipeline-dataset/tensorboard-train',
      }]
    }

    # Exports two sample metrics:
    metrics = {
      'metrics': [{
          'name': 'quotient',
          'numberValue':  float(quotient),
        },{
          'name': 'remainder',
          'numberValue':  float(remainder),
        }]}

    from collections import namedtuple
    divmod_output = namedtuple('MyDivmodOutput', ['quotient', 'remainder', 'mlpipeline_ui_metadata', 'mlpipeline_metrics'])
    return divmod_output(quotient, remainder, json.dumps(metadata), json.dumps(metrics))


# divmod_op = comp.create_component_from_func(my_divmod, base_image='quay.io/hukhan/tensorflow/tensorflow:2.12.0')


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
    # s3_endpoint = os.getenv("S3_ENDPOINT")
    # access_key = os.getenv("ACCESS_KEY")
    # secret_key = os.getenv("SECRET_KEY")
    # env_var_01 = V1EnvVar(name='S3_ENDPOINT', value=s3_endpoint)
    # env_var_02 = V1EnvVar(name='ACCESS_KEY', value=access_key)
    # env_var_03 = V1EnvVar(name='SECRET_KEY', value=secret_key)
    # ingest_task = ingest_op('IBM').add_env_variable(env_var_01).add_env_variable(env_var_02).add_env_variable(env_var_03)
    ingest_task = ingest_op(ticker)\
      .add_env_variable(V1EnvVar(name='S3_ENDPOINT', value=os.getenv('S3_ENDPOINT')))\
      .add_env_variable(V1EnvVar(name='ACCESS_KEY', value=os.getenv('ACCESS_KEY')))\
      .add_env_variable(V1EnvVar(name='SECRET_KEY', value=os.getenv('SECRET_KEY')))

    train_task = train_op(ingest_task.outputs['filename'])

    # Passing a task output reference as operation arguments
    # For an operation with a single return value, the output reference can be accessed using `task.output` or
    # `task.outputs['output_name']` syntax
    # 
    # divmod_task = divmod_op(add_task.output, b)
    # divmod_task.add_pod_annotation("tekton.dev/track_step_artifact", "true")

    # For an operation with a multiple return values, the output references can be accessed using
    # `task.outputs['output_name']` syntax
    # 
    # result_task = add_op(divmod_task.outputs['quotient'], c)


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
