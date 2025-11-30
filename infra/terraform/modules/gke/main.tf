locals {
  node_pool_map = { for pool in var.node_pools : pool.name => pool }
}

resource "google_container_cluster" "this" {
  provider = google-beta
  name     = var.cluster_name
  project  = var.project_id
  location = var.region

  remove_default_node_pool = true
  initial_node_count       = 1
  networking_mode          = "VPC_NATIVE"
  network                  = var.network
  subnetwork               = var.subnetwork
  node_locations           = length(var.zones) > 0 ? var.zones : null

  workload_identity_config {
    workload_pool = var.workload_identity_pool
  }

  ip_allocation_policy {
    cluster_secondary_range_name  = var.pod_secondary_range
    services_secondary_range_name = var.service_secondary_range
  }

  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = var.master_ipv4_cidr
  }

  dynamic "master_authorized_networks_config" {
    for_each = length(var.master_authorized_cidrs) > 0 ? [true] : []
    content {
      dynamic "cidr_blocks" {
        for_each = var.master_authorized_cidrs
        content {
          cidr_block   = cidr_blocks.value.cidr_block
          display_name = try(cidr_blocks.value.description, "")
        }
      }
    }
  }

  release_channel {
    channel = var.release_channel
  }

  vertical_pod_autoscaling {
    enabled = true
  }

  logging_config {
    enable_components = var.logging_components
  }

  monitoring_config {
    enable_components = var.monitoring_components
  }

  addons_config {
    http_load_balancing {
      disabled = false
    }
    horizontal_pod_autoscaling {
      disabled = false
    }
  }

  cluster_autoscaling {
    autoscaling_profile = "OPTIMIZE_UTILIZATION"
  }
}

resource "google_container_node_pool" "this" {
  provider = google-beta
  for_each = local.node_pool_map

  name     = each.value.name
  project  = var.project_id
  cluster  = google_container_cluster.this.name
  location = var.region

  node_config {
    machine_type    = each.value.machine_type
    disk_size_gb    = try(each.value.disk_size_gb, 100)
    disk_type       = try(each.value.disk_type, "pd-balanced")
    service_account = try(each.value.service_account, null)
    spot            = try(each.value.spot, false)
    labels          = merge({ pool = each.value.name }, try(each.value.labels, {}))
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }

    dynamic "taints" {
      for_each = try(each.value.taints, [])
      content {
        key    = taints.value.key
        value  = taints.value.value
        effect = taints.value.effect
      }
    }

    dynamic "guest_accelerator" {
      for_each = try(each.value.gpu != null ? [each.value.gpu] : [], [])
      content {
        type  = guest_accelerator.value.type
        count = guest_accelerator.value.count
      }
    }
  }

  autoscaling {
    min_node_count = each.value.min_count
    max_node_count = each.value.max_count
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  initial_node_count = max(each.value.min_count, 1)
}
