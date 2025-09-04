Deployment (`src/deployment`)
=============================

This module enables deployment of the emotion classifier to cloud environments such as Azure or Kubernetes.

Azure ML Deployment
-------------------

.. automodule:: src.deployment.cloud_deploy
   :members:
   :undoc-members:
   :show-inheritance:

- Uses Azure ML SDK to register and deploy the model
- Submits score script and environment YAML

Kubernetes Endpoint Deployment
------------------------------

.. automodule:: src.deployment.deploy_kubernetes_endpoint
   :members:
   :undoc-members:
   :show-inheritance:

- Wraps model in a FastAPI container
- Prepares for deployment to AKS, GKE, or other Kubernetes clusters

Score Script
------------

.. automodule:: src.deployment.score
   :members:
   :undoc-members:
   :show-inheritance:

Required for Azure ML endpoints. Loads the model and returns predictions based on input schema.

File Structure
--------------

::

    src/
    └── deployment/
        ├── cloud_deploy.py
        ├── deploy_kubernetes_endpoint.py
        ├── score.py
        └── deployment_README.md
