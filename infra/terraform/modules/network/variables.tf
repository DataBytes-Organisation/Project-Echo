variable "project_id" {
  type        = string
  description = "Google Cloud project identifier"
}

variable "network_name" {
  type        = string
  description = "VPC network name"
}

variable "subnet_name" {
  type        = string
  description = "Primary subnetwork name"
}

variable "region" {
  type        = string
  description = "Region for subnetwork and router"
}

variable "primary_cidr" {
  type        = string
  description = "CIDR block for the primary subnetwork"
}

variable "pod_cidr" {
  type        = string
  description = "Secondary CIDR block for pod IPs"
}

variable "service_cidr" {
  type        = string
  description = "Secondary CIDR block for service IPs"
}
