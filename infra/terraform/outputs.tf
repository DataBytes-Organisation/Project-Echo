output "cluster_name" {
  value       = module.gke.cluster_name
  description = "Name of the managed GKE cluster"
}

output "cluster_endpoint" {
  value       = module.gke.endpoint
  description = "Public endpoint for the GKE control plane"
}

output "network_name" {
  value       = module.network.network_name
  description = "Name of the provisioned VPC network"
}

output "artifact_registry_repository" {
  value       = google_artifact_registry_repository.primary.repository_id
  description = "Artifact Registry repository identifier"
}

output "workload_service_accounts" {
  value       = { for key, sa in google_service_account.workloads : key => sa.email }
  description = "Map of workload service account keys to emails"
}
