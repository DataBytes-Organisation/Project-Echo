variable "project_id" {}
variable "region" {}
variable "zones" { type = list(string) }
variable "cluster_name" { default = "echonet-gke" }
variable "model_buckets" { type = list(string) }
variable "artifact_repo_name" { default = "echonet" }
variable "model_bucket_stg" {}
variable "model_bucket_prod" {}
variable "api_sa_id" { default = "api-sa" }
variable "engine_sa_id" { default = "engine-sa" }
variable "model_sa_id" { default = "model-sa" }
variable "secrets" { type = list(string) default = ["mongo-uri", "twilio-auth-token"] }
