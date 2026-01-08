output "cluster_name" {
  value       = google_container_cluster.this.name
  description = "Name of the created cluster"
}

output "endpoint" {
  value       = google_container_cluster.this.endpoint
  description = "Endpoint of the GKE API server"
}

output "node_pool_ids" {
  value       = [for pool in google_container_node_pool.this : pool.id]
  description = "Identifiers of the managed node pools"
}
