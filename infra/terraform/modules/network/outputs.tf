output "network_name" {
  value       = google_compute_network.this.name
  description = "Name of the created VPC network"
}

output "network_self_link" {
  value       = google_compute_network.this.self_link
  description = "Self link of the VPC network"
}

output "subnet_name" {
  value       = google_compute_subnetwork.primary.name
  description = "Name of the primary subnetwork"
}

output "subnet_self_link" {
  value       = google_compute_subnetwork.primary.self_link
  description = "Self link of the primary subnetwork"
}

output "pod_secondary_range" {
  value       = local.pod_range_name
  description = "Name of the secondary IP range allocated for pods"
}

output "service_secondary_range" {
  value       = local.service_range_name
  description = "Name of the secondary IP range allocated for services"
}
