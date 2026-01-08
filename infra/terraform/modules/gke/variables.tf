variable "project_id" {
  type        = string
  description = "Google Cloud project identifier"
}

variable "cluster_name" {
  type        = string
  description = "Name of the GKE cluster"
}

variable "region" {
  type        = string
  description = "Region for the regional GKE cluster"
}

variable "zones" {
  type        = list(string)
  default     = []
  description = "Optional list of zones for node pool placement"
}

variable "network" {
  type        = string
  description = "Self link of the VPC network"
}

variable "subnetwork" {
  type        = string
  description = "Self link of the subnetwork"
}

variable "subnet_name" {
  type        = string
  description = "Name of the subnetwork"
}

variable "pod_secondary_range" {
  type        = string
  description = "Secondary IP range name for pods"
}

variable "service_secondary_range" {
  type        = string
  description = "Secondary IP range name for services"
}

variable "workload_identity_pool" {
  type        = string
  description = "Workload Identity pool name"
}

variable "master_ipv4_cidr" {
  type        = string
  description = "Control plane CIDR block"
}

variable "master_authorized_cidrs" {
  type = list(object({
    cidr_block  = string
    description = optional(string, "")
  }))
  default     = []
  description = "CIDR ranges allowed to reach the control plane"
}

variable "release_channel" {
  type        = string
  description = "Desired GKE release channel"
}

variable "logging_components" {
  type        = list(string)
  description = "Logging components to enable"
}

variable "monitoring_components" {
  type        = list(string)
  description = "Monitoring components to enable"
}

variable "node_pools" {
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
  description = "Node pool definitions"
}

