variable "project_id" {
  type        = string
  description = "Google Cloud project identifier"
}

variable "region" {
  type        = string
  description = "Primary deployment region"
}

variable "zones" {
  type        = list(string)
  default     = []
  description = "Optional list of GCP zones for node placement"
}

variable "environment" {
  type        = string
  default     = "dev"
  description = "Environment name used for resource naming"
}

variable "network_name" {
  type        = string
  default     = "echonet-network"
  description = "VPC network name"
}

variable "subnet_name" {
  type        = string
  default     = "echonet-subnet"
  description = "Primary subnetwork name"
}

variable "subnet_ip_cidr" {
  type        = string
  default     = "10.30.0.0/20"
  description = "CIDR for the primary subnetwork"
}

variable "pods_secondary_cidr" {
  type        = string
  default     = "10.40.0.0/16"
  description = "CIDR range allocated for GKE pods"
}

variable "services_secondary_cidr" {
  type        = string
  default     = "10.50.0.0/20"
  description = "CIDR range allocated for GKE services"
}

variable "cluster_name" {
  type        = string
  default     = "echonet-gke"
  description = "Name of the GKE cluster"
}

variable "master_ipv4_cidr_block" {
  type        = string
  default     = "172.16.0.0/28"
  description = "Control plane CIDR block for private GKE clusters"
}

variable "master_authorized_cidrs" {
  type = list(object({
    cidr_block  = string
    description = optional(string, "")
  }))
  default     = []
  description = "CIDR blocks allowed to access the GKE control plane"
}

variable "cluster_release_channel" {
  type        = string
  default     = "REGULAR"
  description = "GKE release channel"
}

variable "cluster_logging_components" {
  type        = list(string)
  default     = ["SYSTEM_COMPONENTS", "WORKLOADS"]
  description = "GKE logging components to enable"
}

variable "cluster_monitoring_components" {
  type        = list(string)
  default     = ["SYSTEM_COMPONENTS", "POD"]
  description = "GKE monitoring components to enable"
}

variable "node_pools" {
  description = "Node pool definitions for the GKE cluster"
  type = list(object({
    name            = string
    machine_type    = string
    min_count       = number
    max_count       = number
    disk_size_gb    = optional(number, 100)
    disk_type       = optional(string, "pd-balanced")
    service_account = optional(string)
    spot            = optional(bool, false)
    labels          = optional(map(string), {})
    taints          = optional(list(object({ key = string, value = string, effect = string })), [])
    gpu             = optional(object({ type = string, count = number }))
  }))
  default = [
    {
      name         = "general"
      machine_type = "e2-standard-4"
      min_count    = 1
      max_count    = 4
      labels       = { role = "general" }
    },
    {
      name         = "gpu"
      machine_type = "n1-standard-4"
      min_count    = 0
      max_count    = 2
      labels       = { role = "gpu" }
      taints       = [{ key = "gpu", value = "true", effect = "NO_SCHEDULE" }]
      gpu          = { type = "nvidia-tesla-t4", count = 1 } # Guardrails restrict to N1 + Tesla T4, max 2 GPUs/project
    }
  ]
}

variable "artifact_repo_name" {
  type        = string
  default     = "echonet"
  description = "Artifact Registry repository name"
}

variable "artifact_repo_location" {
  type        = string
  default     = null
  description = "Location for Artifact Registry (defaults to region)"
}

variable "model_buckets" {
  description = "Model storage buckets keyed by environment"
  type = map(object({
    name           = string
    location       = string
    retention_days = number
    storage_class  = optional(string, "STANDARD")
    force_destroy  = optional(bool, false)
    labels         = optional(map(string), {})
  }))
  default = {}
}

variable "workload_service_accounts" {
  description = "Workload service account definitions"
  type = map(object({
    display_name = string
    description  = optional(string)
    roles        = list(string)
  }))
  default = {
    api = {
      display_name = "API Workload"
      roles = [
        "roles/artifactregistry.reader",
        "roles/secretmanager.secretAccessor",
        "roles/storage.objectViewer"
      ]
    }
    engine = {
      display_name = "Engine Workload"
      roles = [
        "roles/secretmanager.secretAccessor",
        "roles/storage.objectViewer"
      ]
    }
    hmi = {
      display_name = "HMI Workload"
      roles = [
        "roles/secretmanager.secretAccessor"
      ]
    }
  }
}

variable "workload_identity_bindings" {
  description = "Mappings between Google and Kubernetes service accounts"
  type = list(object({
    service_account = string
    namespace       = string
    ksa             = string
  }))
  default = [
    {
      service_account = "api"
      namespace       = "project-echo"
      ksa             = "api"
    },
    {
      service_account = "engine"
      namespace       = "project-echo"
      ksa             = "engine"
    },
    {
      service_account = "hmi"
      namespace       = "project-echo"
      ksa             = "hmi"
    }
  ]
}

variable "secret_names" {
  type        = list(string)
  default     = ["mongo-uri", "redis-password", "twilio-auth-token"]
  description = "Secret Manager secret identifiers to provision"
}

variable "default_labels" {
  type        = map(string)
  default     = {}
  description = "Additional labels applied to managed resources"
}

variable "project_services_additional" {
  type        = list(string)
  default     = []
  description = "Additional APIs to enable on the project"
}
