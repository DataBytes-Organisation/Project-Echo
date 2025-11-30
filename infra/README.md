# Cloud Deployment Guide

This folder contains the infrastructure-as-code used to run Project Echo on Google Cloud Platform.

## Terraform (`infra/terraform`)

1. Copy `terraform.tfvars.example` to `terraform.tfvars` and update the values for your project (project id, regions, bucket names, Workload Identity bindings, etc.).
2. (Optional) Configure remote state in `backend.tf` if you do not want to use the default local state.
3. Initialise and review the plan:
   ```bash
   cd infra/terraform
   terraform init
   terraform plan
   ```
4. Apply when satisfied:
   ```bash
   terraform apply -auto-approve
   ```

The Terraform stack enables the required APIs, provisions a private GKE cluster, creates service accounts + Workload Identity bindings, Artifact Registry, Secret Manager entries, Cloud NAT, and model storage buckets.

## Kubernetes (`k8s/`)

`k8s/base` defines the shared manifests for the API, inference engine, HMI, MongoDB, and Redis components. The `k8s/overlays/dev` and `k8s/overlays/prod` folders provide environment-specific adjustments. Update the image registries, service-account annotations, ingress hosts, and secret literals before deploying.

Render and apply an overlay:
```bash
kubectl apply -k k8s/overlays/dev
```

## Cloud Build (`cloudbuild.yaml`)

`cloudbuild.yaml` builds the API, engine, and HMI containers, pushes them to Artifact Registry, and deploys the selected overlay. Adjust `_ENV`, `_REGION`, `_REPOSITORY`, and `_CLUSTER` substitutions to match your environment or override them when triggering the build.
