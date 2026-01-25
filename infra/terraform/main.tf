locals {
  artifact_repo_location = coalesce(var.artifact_repo_location, var.region)

  service_account_role_pairs = flatten([
    for sa_key, sa_def in var.workload_service_accounts : [
      for role in sa_def.roles : {
        key    = "${sa_key}-${replace(role, "/", "-")}"
        sa_key = sa_key
        role   = role
      }
    ]
  ])

  bucket_config = {
    for env, bucket in var.model_buckets : env => {
      name           = replace(bucket.name, "PROJECT_ID", var.project_id)
      location       = bucket.location
      retention_days = bucket.retention_days
      storage_class  = try(bucket.storage_class, "STANDARD")
      labels         = merge({ env = env, app = "echonet" }, var.default_labels, try(bucket.labels, {}))
      force_destroy  = try(bucket.force_destroy, false)
    }
  }
}

module "network" {
  # Required GCP services (compute, container, servicenetworking, etc.) must be pre-enabled by an owner.
  source       = "./modules/network"
  project_id   = var.project_id
  network_name = var.network_name
  subnet_name  = var.subnet_name
  region       = var.region
  primary_cidr = var.subnet_ip_cidr
  pod_cidr     = var.pods_secondary_cidr
  service_cidr = var.services_secondary_cidr
}

module "gke" {
  source                  = "./modules/gke"
  providers               = { google-beta = google-beta }
  project_id              = var.project_id
  cluster_name            = var.cluster_name
  region                  = var.region
  zones                   = var.zones
  network                 = module.network.network_self_link
  subnetwork              = module.network.subnet_self_link
  subnet_name             = module.network.subnet_name
  pod_secondary_range     = module.network.pod_secondary_range
  service_secondary_range = module.network.service_secondary_range
  workload_identity_pool  = "${var.project_id}.svc.id.goog"
  master_ipv4_cidr        = var.master_ipv4_cidr_block
  master_authorized_cidrs = var.master_authorized_cidrs
  release_channel         = var.cluster_release_channel
  logging_components      = var.cluster_logging_components
  monitoring_components   = var.cluster_monitoring_components
  node_pools              = var.node_pools
  depends_on = [module.network]
}

resource "google_artifact_registry_repository" "primary" {
  project       = var.project_id
  location      = local.artifact_repo_location
  repository_id = var.artifact_repo_name
  format        = "DOCKER"
  description   = "Container images for Project Echo workloads"
  labels        = merge({ app = "echonet" }, var.default_labels)
}

resource "google_storage_bucket" "models" {
  for_each                    = local.bucket_config
  name                        = each.value.name
  location                    = each.value.location
  storage_class               = each.value.storage_class
  force_destroy               = each.value.force_destroy
  uniform_bucket_level_access = true
  versioning { enabled = true }
  lifecycle_rule {
    action { type = "Delete" }
    condition { age = each.value.retention_days }
  }
  labels = each.value.labels
}

data "google_service_account" "workloads" {
  for_each = var.workload_service_accounts
  project  = var.project_id
  # Service accounts must be pre-created by a project owner using this naming pattern.
  account_id = substr(
    lower(replace("${var.environment}-${each.key}", "_", "-")),
    0,
    30
  )
}

resource "google_project_iam_member" "workload_roles" {
  for_each = { for item in local.service_account_role_pairs : item.key => item }
  project  = var.project_id
  role     = each.value.role
  member   = "serviceAccount:${data.google_service_account.workloads[each.value.sa_key].email}"
}

resource "google_service_account_iam_member" "workload_identity" {
  for_each           = { for binding in var.workload_identity_bindings : "${binding.service_account}-${binding.namespace}-${binding.ksa}" => binding }
  service_account_id = data.google_service_account.workloads[each.value.service_account].name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[${each.value.namespace}/${each.value.ksa}]"
}

resource "google_secret_manager_secret" "managed" {
  for_each  = toset(var.secret_names)
  secret_id = each.value
  replication {
    user_managed {
      replicas {
        location = var.region
      }
    }
  }
  labels = merge({ app = "echonet" }, var.default_labels)
}
