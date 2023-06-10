# Experiment with predicting stock prices using Long Short-Term Memory (LSTM)

### Prerequisites
- An Openshift 4.11+ cluster
- The Pipelines Operator
- The `oc` and `tkn` command line tools (See the question mark menu in the Openshift UI)

### Files and directories
```
├── src                             Python source for data ingestion and model training
├── pipelines                       Tekton pipeline and tasks 
├── data                            Sample data
├── notebooks                       Jupyter experimentation
└── requirements.txt                Python dependencies
```

### Pipeline example
```
cd pipelines
```

Apply the custom tasks and pipeline resources.
```
oc apply -f pipelines/01-ingest-train-task.yaml
oc apply -f pipelines/02-ingest-train-pipeline.yaml
```

-> Use the Openshift UI to manually create a storage persistent volume claim (PVC) and 
pass its name in when starting the pipeline below.

### Starting a pipeline run
```
tkn pipeline start ingest-and-train -w name=shared-workspace,claimName=my-pipeline-claim-01 -p deployment-name=ingest-and-train -p git-url=https://github.com/bkoz/stock.git -p IMAGE='image-registry.openshift-image-registry.svc:5000/$(context.pipelineRun.namespace)/ingest-and-train' --use-param-defaults
```

Option to auto-create a pvc when starting the pipeline.
```
tkn pipeline start ingest-and-train -w name=shared-workspace,volumeClaimTemplateFile=00_persistent_volume_claim.yaml -p deployment-name=ingest-and-train -p git-url=https://github.com/bkoz/stock.git -p IMAGE='image-registry.openshift-image-registry.svc:5000/$(context.pipelineRun.namespace)/ingest-and-train' --use-param-defaults
```

### References
[Data streamer sample](https://github.com/redhat-na-ssa/ml_data_streamer/blob/main/source-eip/src/test/resources/samples/MUFG-1.csv)

[Custom Notebook Builder](https://github.com/redhat-na-ssa/rhods-custom-notebook-example.git)

[Pipeline examples](https://github.com/rh-datascience-and-edge-practice/kubeflow-examples/blob/main/pipelines/11_iris_training_pipeline.py)
