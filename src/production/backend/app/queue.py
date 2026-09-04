import os
import logging
from redis import Redis
from rq import Queue

logger = logging.getLogger(__name__)

# Initialize Redis connection using db=1 for the job queue to avoid conflicts with HMI sessions
REDIS_HOST = os.getenv("REDIS_HOST", "echo-redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# We use decode_responses=False for RQ compatibility
redis_conn = Redis(host=REDIS_HOST, port=REDIS_PORT, db=1, decode_responses=False)
job_queue = Queue("echo-backend", connection=redis_conn)

def dummy_background_job(data: dict):
    """
    A simple background job to verify the queue is functioning.
    """
    logger.info(f"Processing dummy job with data: {data}")
    return True
