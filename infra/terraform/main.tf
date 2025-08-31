terraform {
  required_version = ">= 1.6.0"
  required_providers {
    google = { source = "hashicorp/google" version = "~> 5.0" }
  }
}
provider "google" {
  project = var.project_id
  region  = var.region
}
resource "google_container_cluster" "echonet" {
  name                     = var.cluster_name
  location                 = var.region
  remove_default_node_pool = true
  initial_node_count       = 1
  networking_mode          = "VPC_NATIVE"
  workload_identity_config { workload_pool = "${var.project_id}.svc.id.goog" }
  release_channel { channel = "REGULAR" }
}
resource "google_container_node_pool" "general" {
  name       = "general-pool"
  cluster    = google_container_cluster.echonet.name
  location   = var.region
  node_config {
    machine_type = "e2-standard-4"
    oauth_scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    labels = { role = "general" }
  }
  initial_node_count = 2
  autoscaling { min_node_count = 1 max_node_count = 4 }
}
resource "google_container_node_pool" "gpu" {
  name     = "gpu-pool"
  cluster  = google_container_cluster.echonet.name
  location = var.region
  node_config {
    machine_type = "g2-standard-4"
    guest_accelerator { type = "nvidia-l4" count = 1 }
    labels = { role = "gpu" }
    taints = [{ key = "gpu", value = "true", effect = "NO_SCHEDULE" }]
  }
  initial_node_count = 1
  autoscaling { min_node_count = 0 max_node_count = 3 }
}
resource "google_artifact_registry_repository" "echonet" {
  location      = var.region
  repository_id = var.artifact_repo_name
  description   = "EchoNet container images"
  format        = "DOCKER"
}

resource "google_storage_bucket" "model_stg" {
  name          = replace(var.model_bucket_stg, "gs://", "")
  location      = var.region
  force_destroy = false
  uniform_bucket_level_access = true
  versioning { enabled = true }
  lifecycle_rule { action { type = "Delete" } condition { age = 120 } }
  labels = { env = "staging", app = "echonet" }
}
resource "google_storage_bucket" "model_prod" {
  name          = replace(var.model_bucket_prod, "gs://", "")
  location      = var.region
  force_destroy = false
  uniform_bucket_level_access = true
  versioning { enabled = true }
  lifecycle_rule { action { type = "Delete" } condition { age = 365 } }
  labels = { env = "prod", app = "echonet" }
}

# Service accounts for workload identity
resource "google_service_account" "api" { account_id = var.api_sa_id display_name = "API Workload" }
resource "google_service_account" "engine" { account_id = var.engine_sa_id display_name = "Engine Workload" }
resource "google_service_account" "model" { account_id = var.model_sa_id display_name = "Model Server Workload" }

# Secret Manager secrets (placeholders)
resource "google_secret_manager_secret" "secrets" {
  for_each  = toset(var.secrets)
  secret_id = each.key
  replication { automatic = true }
}
