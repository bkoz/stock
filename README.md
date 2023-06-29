# Experiment with predicting stock prices using Long Short-Term Memory (LSTM)

## Prerequisites
- An Openshift 4.11+ cluster
- The Openshift Pipelines Operator
- The `oc` and `tkn` command line tools (See the question mark menu in the Openshift UI)
- An Openshift project to work with

The version of `kfp-tekton` is important for launching pipeline runs
via the python sdk against the Openshift Kubeflow Pipeline API Server.
```
kfp                      1.8.19
kfp-pipeline-spec        0.1.16
kfp-server-api           1.8.5
kfp-tekton               1.5.2
kfp-tekton-server-api    1.5.0
```

### Files and directories
```
├── src                             Python source for data ingestion and model training
├── pipelines                       Tekton pipeline and tasks 
├── data                            Sample data
├── notebooks                       Jupyter experimentation
└── requirements.txt                Python dependencies
```

### Environment Variables (.env file)
```

```
### Pipeline example
```
python 01-ingest-train.py
```

### Monitoring pipeline runs
```
tkn pipelineruns list
```


### References
[Trevor's examples](https://github.com)
