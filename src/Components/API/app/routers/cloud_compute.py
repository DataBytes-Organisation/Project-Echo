"""
Cloud Compute Router
Provides endpoints for fetching Google Cloud Platform information
"""

from fastapi import APIRouter, HTTPException
from google.cloud import compute_v1
from google.cloud import billing_v1
from google.cloud import bigquery
from google.cloud import monitoring_v3
from typing import List, Dict, Any
from datetime import datetime, timedelta
import calendar
import os

router = APIRouter()

def get_billing_account_name(project_id: str) -> str:
    """Get the billing account name for a project"""
    try:
        client = billing_v1.CloudBillingClient()
        project_billing_info = client.get_project_billing_info(name=f"projects/{project_id}")
        return project_billing_info.billing_account_name
    except Exception as e:
        print(f"Error getting billing account: {e}")
        return None

def get_cpu_utilization(project_id: str, zone: str = "australia-southeast1-a") -> float:
    """Get average CPU utilization for all instances in the project"""
    try:
        client = monitoring_v3.MetricServiceClient()
        project_name = f"projects/{project_id}"
        
        # Query for last 5 minutes of CPU data
        now = datetime.utcnow()
        interval = monitoring_v3.TimeInterval({
            "end_time": {"seconds": int(now.timestamp())},
            "start_time": {"seconds": int((now - timedelta(minutes=5)).timestamp())},
        })
        
        # CPU utilization metric
        results = client.list_time_series(
            request={
                "name": project_name,
                "filter": 'metric.type="compute.googleapis.com/instance/cpu/utilization"',
                "interval": interval,
                "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
            }
        )
        
        # Average all instances' CPU usage
        cpu_values = []
        for result in results:
            for point in result.points:
                cpu_values.append(point.value.double_value * 100)  # Convert to percentage
        
        if cpu_values:
            return round(sum(cpu_values) / len(cpu_values), 1)
        return None
        
    except Exception as e:
        print(f"Error fetching CPU metrics: {e}")
        return None

def get_disk_utilization(project_id: str, zone: str = "australia-southeast1-a") -> dict:
    """Get disk utilization for all instances in the project"""
    try:
        # Try to get disk metrics from agent
        client = monitoring_v3.MetricServiceClient()
        project_name = f"projects/{project_id}"
        
        now = datetime.utcnow()
        interval = monitoring_v3.TimeInterval({
            "end_time": {"seconds": int(now.timestamp())},
            "start_time": {"seconds": int((now - timedelta(minutes=5)).timestamp())},
        })
        
        # Disk utilization metric (percentage)
        results = client.list_time_series(
            request={
                "name": project_name,
                "filter": 'metric.type="agent.googleapis.com/disk/percent_used"',
                "interval": interval,
                "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
            }
        )
        
        disk_values = []
        for result in results:
            for point in result.points:
                disk_values.append(point.value.double_value)
        
        if disk_values:
            avg_percent = round(sum(disk_values) / len(disk_values), 1)
            # Get disk size from instance
            compute_client = compute_v1.InstancesClient()
            instances = list(compute_client.list(project=project_id, zone=zone))
            
            total_gb = 30  # Default boot disk size
            if instances:
                for disk in instances[0].disks:
                    if disk.boot:
                        total_gb = disk.disk_size_gb
                        break
            
            used_gb = round((avg_percent / 100) * total_gb, 1)
            return {
                "used_gb": used_gb,
                "total_gb": total_gb,
                "percent": avg_percent
            }
        
        # Fallback: Get disk size from instance configuration
        compute_client = compute_v1.InstancesClient()
        instances = list(compute_client.list(project=project_id, zone=zone))
        
        if instances:
            total_gb = 30  # Default
            for disk in instances[0].disks:
                if disk.boot:
                    total_gb = disk.disk_size_gb
                    break
            
            # Estimate ~40% usage without agent
            estimated_used = round(total_gb * 0.40, 1)
            return {
                "used_gb": estimated_used,
                "total_gb": total_gb,
                "percent": 40.0,
                "estimated": True
            }
        
        return None
        
    except Exception as e:
        print(f"Error fetching disk metrics: {e}")
        return None

def get_memory_utilization(project_id: str, zone: str = "australia-southeast1-a") -> dict:
    """Get memory utilization for all instances in the project"""
    try:
        # First try agent memory metrics
        client = monitoring_v3.MetricServiceClient()
        project_name = f"projects/{project_id}"
        
        # Query for last 5 minutes of memory data
        now = datetime.utcnow()
        interval = monitoring_v3.TimeInterval({
            "end_time": {"seconds": int(now.timestamp())},
            "start_time": {"seconds": int((now - timedelta(minutes=5)).timestamp())},
        })
        
        # Memory utilization metric (percentage)
        results = client.list_time_series(
            request={
                "name": project_name,
                "filter": 'metric.type="agent.googleapis.com/memory/percent_used" AND metric.label.state="used"',
                "interval": interval,
                "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
            }
        )
        
        # Average memory usage percentage
        memory_values = []
        for result in results:
            for point in result.points:
                memory_values.append(point.value.double_value)
        
        if memory_values:
            avg_percent = round(sum(memory_values) / len(memory_values), 1)
            # Get actual instance memory from machine type
            compute_client = compute_v1.InstancesClient()
            instances = compute_client.list(project=project_id, zone=zone)
            
            total_gb = 16  # Default
            for instance in instances:
                machine_type = instance.machine_type.split('/')[-1]
                # Parse memory from machine type (e.g., e2-medium = 4GB, n1-standard-1 = 3.75GB)
                if 'e2-medium' in machine_type:
                    total_gb = 4
                elif 'e2-small' in machine_type:
                    total_gb = 2
                elif 'n1-standard-1' in machine_type:
                    total_gb = 3.75
                elif 'n2-standard-2' in machine_type:
                    total_gb = 8
                break
            
            used_gb = round((avg_percent / 100) * total_gb, 1)
            return {
                "used_gb": used_gb,
                "total_gb": total_gb,
                "percent": avg_percent
            }
        
        # Fallback: Try to get instance info and show based on machine type
        compute_client = compute_v1.InstancesClient()
        instances = list(compute_client.list(project=project_id, zone=zone))
        
        if instances:
            machine_type = instances[0].machine_type.split('/')[-1]
            total_gb = 4  # Default for e2-medium
            
            # Estimate memory based on machine type
            if 'e2-medium' in machine_type:
                total_gb = 4
            elif 'e2-small' in machine_type:
                total_gb = 2
            elif 'n1-standard-1' in machine_type:
                total_gb = 3.75
            elif 'n2-standard-2' in machine_type:
                total_gb = 8
                
            # Without agent, estimate ~30-40% usage
            estimated_used = round(total_gb * 0.35, 1)
            return {
                "used_gb": estimated_used,
                "total_gb": total_gb,
                "percent": 35.0,
                "estimated": True
            }
        
        return None
        
    except Exception as e:
        print(f"Error fetching memory metrics: {e}")
        return None

@router.get("/cloud-info")
async def get_cloud_info():
    """
    Fetch Google Cloud Compute Engine instance details
    
    Returns:
        dict: Dictionary containing list of instances with their details
    """
    try:
        # Initialize Google Cloud client lazily
        client = compute_v1.InstancesClient()
        
        # Google Cloud project configuration
        project = "sit-23t1-project-echo-25288b9"
        zone = "australia-southeast1-b"

        # Fetch instance details
        instances = client.list(project=project, zone=zone)
        instance_list = [
            {
                "name": instance.name,
                "status": instance.status,
                "machine_type": instance.machine_type,
                "zone": zone,
            }
            for instance in instances
        ]

        return {
            "success": True,
            "instances": instance_list,
            "count": len(instance_list)
        }
    except Exception as e:
        # Return dummy data if credentials are not available
        return {
            "success": False,
            "message": "Using dummy data - Google Cloud credentials not configured",
            "instances": [
                {
                    "name": "instance-1",
                    "status": "RUNNING",
                    "machine_type": "e2-medium",
                    "zone": "us-central1-a",
                },
                {
                    "name": "instance-2",
                    "status": "RUNNING",
                    "machine_type": "n1-standard-1",
                    "zone": "us-central1-a",
                }
            ],
            "count": 2,
            "error": str(e)
        }

@router.get("/cloud-metrics")
async def get_cloud_metrics():
    """
    Fetch aggregated cloud metrics including real-time billing information
    
    Returns:
        dict: Dictionary containing CPU, memory, storage, and billing metrics
    """
    try:
        project_id = "sit-23t1-project-echo-25288b9"

        # Fetch real CPU utilization from Cloud Monitoring
        cpu_usage = get_cpu_utilization(project_id)
        if cpu_usage is None:
            cpu_usage = 45  # Fallback value

        # Fetch real memory utilization from Cloud Monitoring
        memory_data = get_memory_utilization(project_id)
        if memory_data:
            memory_usage = f"{memory_data['used_gb']} / {memory_data['total_gb']} GB"
            memory_percent = memory_data['percent']
        else:
            memory_usage = "6 / 16 GB"  # Fallback value
            memory_percent = 37.5

        # Fetch real disk utilization from Cloud Monitoring
        disk_data = get_disk_utilization(project_id)
        if disk_data:
            storage_usage = f"{disk_data['used_gb']} / {disk_data['total_gb']} GB"
        else:
            storage_usage = "1.3 / 3 TB"  # Fallback value

        # Attempt to get billing account (for display)
        billing_account = get_billing_account_name(project_id)

        # Try querying BigQuery billing export if configured
        dataset = os.getenv("BILLING_DATASET")
        table = os.getenv("BILLING_TABLE")  # e.g., gcp_billing_export_v1
        bq_project = os.getenv("BILLING_PROJECT_ID", project_id)

        month_cost = None
        forecast_cost = None
        currency = "AUD"
        last_updated = None

        if dataset and table:
            try:
                client = bigquery.Client(project=bq_project)

                # Compute current month window
                now = datetime.utcnow()
                start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                days_in_month = calendar.monthrange(now.year, now.month)[1]
                day_of_month = now.day

                query = (
                    f"""
                    SELECT
                      ROUND(SUM(cost), 2) AS month_to_date_cost,
                      ANY_VALUE(currency) AS currency,
                      MAX(usage_end_time) AS last_usage_time
                    FROM `{bq_project}.{dataset}.{table}`
                    WHERE project.id = @project_id
                      AND usage_start_time >= @start
                      AND usage_start_time < @end
                    """
                )

                job_config = bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter("project_id", "STRING", project_id),
                        bigquery.ScalarQueryParameter("start", "TIMESTAMP", start_of_month),
                        bigquery.ScalarQueryParameter("end", "TIMESTAMP", now),
                    ]
                )

                result = client.query(query, job_config=job_config).result()
                for row in result:
                    month_cost = float(row["month_to_date_cost"]) if row["month_to_date_cost"] is not None else 0.0
                    currency = row["currency"] or currency
                    last_updated = row["last_usage_time"].isoformat() if row["last_usage_time"] else None

                if month_cost is not None:
                    # Simple forecast: linear projection based on average daily spend
                    avg_per_day = month_cost / max(day_of_month, 1)
                    forecast_val = round(avg_per_day * days_in_month, 2)
                    forecast_cost = forecast_val

            except Exception as bq_err:
                print(f"BigQuery billing query error: {bq_err}")

        # Build response, using live values if available, else fallback placeholders
        current_cost_str = (
            f"${month_cost:.2f} {currency}" if month_cost is not None else "$61.97 AUD"
        )
        forecast_cost_str = (
            f"${forecast_cost:.2f} {currency}" if forecast_cost is not None else "$645.00 AUD"
        )

        payload = {
            "success": True if month_cost is not None else False,
            "message": None if month_cost is not None else "Billing export not configured; using placeholders",
            "metrics": {
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "storage_usage": storage_usage,
                "cpu_trend": "Rising" if cpu_usage and cpu_usage > 50 else "Stable",
                "cost_current": current_cost_str,
                "cost_forecast": forecast_cost_str,
                "billing_account": billing_account,
                "last_updated": last_updated,
            },
        }

        return payload

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching cloud metrics: {str(e)}")
