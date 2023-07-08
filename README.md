# Experiment with predicting stock prices using Long Short-Term Memory (LSTM)

## Prerequisites
- An Openshift 4.11+ cluster
- The Openshift Pipelines Operator
- The `oc` and `tkn` command line tools (See the question mark menu in the Openshift UI)
- An Openshift project to work with
- S3 compatible object storage
- Python 3.9

### Files and Directories
```
.
├── Dockerfile                    Recipe to the pipeline stage container image
├── data
│   ├── IBM.csv                   Example data for local dev/testing
├── notebooks                     Notebook version of the KubeFlow DSL
├── requirements.txt              Python requirements
└── src
    └── 01-ingest-train-kfp.py    KubeFlow DSL
```
### Openshift Object Storage
- Needed to support pipeline parameter passing between stages.

```
export DEFAULT_STORAGE_CLASS=gp3-csi
export DEFAULT_STORAGE_SIZE=2Gi
export DEFAULT_ACCESSMODES=ReadWriteOnce
```

I'm thinking the `openshift-storage.noobaa.io` storage class should also work.

The version of `kfp-tekton` is important for launching pipeline runs
via the python sdk against the Openshift Kubeflow Pipeline API Server.
```
kfp                      1.8.19
kfp-pipeline-spec        0.1.16
kfp-server-api           1.8.5
kfp-tekton               1.5.2
kfp-tekton-server-api    1.5.0
```

### Containers
Build and push the pipeline stage container image to your favorite registry. 
```
podman build -t pipestage .
podman tag localhost/pipestage quay.io/my-user/pipestage
podman push quay.io/my-user/pipestage
```

### Files and directories
```
├── src                             Python source for data ingestion and model training
├── pipelines                       Tekton pipeline and tasks 
├── data                            Sample data
├── notebooks                       Jupyter experimentation
└── requirements.txt                Python dependencies
```

### Environment Variables Example (.env file)
```
ACCESS_KEY=<s3-access-key>
SECRET_KEY=<s3-secret=key>
S3_ENDPOINT=minio-<namespace>.apps.ocp.sandbox1234.opentlc.com
KUBEFLOW_ENDPOINT=https://ds-pipeline-pipelines-definition-<namespace>.apps.ocp.sandbox1234.opentlc.com
BEARER_TOKEN=<oc_whoami_--show-token>
DEFAULT_STORAGE_CLASS=gp3-csi
DEFAULT_STORAGE_SIZE=2Gi
DEFAULT_ACCESSMODES=ReadWriteOnce
```
### Pipeline example
```
python src/01-ingest-train.py
```

### Monitoring pipeline runs
```
tkn pipelineruns list
tkn pipelineruns logs <run-id>
```

### Storage 
```
s3cmd --access_key=<> --secret_key=<> --host=minio-pipelines-tutorial.apps.ocp.sandbox2000.opentlc.com ls --recursive s3://
```
### References
[Trevor's examples](https://github.com/rh-datascience-and-edge-practice/kubeflow-examples.git)
