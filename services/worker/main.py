"""Simplified LangChain-powered ingestion worker."""
import logging
import time
from typing import Any, Tuple

import pika
import pika.exceptions
from dotenv import load_dotenv

from app.config import settings
from app.health import health_status, start_health_server
from app.storage import verify_storage_health
from app.worker import process_article_job

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def verify_environment() -> bool:
    """Check that required services and config are available."""
    logger.info("Verifying environment...")
    
    # Check if required settings have values
    if not settings.rabbitmq_uri.strip():
        logger.error("ERROR: RABBITMQ_URI is empty")
        return False
        
    if not settings.chroma_host.strip():
        logger.error("ERROR: CHROMA_HOST is empty")
        return False
    
    # Check API key for embeddings
    if not settings.openai_api_key and not (settings.azure_openai_enabled and settings.azure_openai_api_key):
        logger.error("ERROR: No OpenAI or Azure OpenAI credentials configured")
        return False
    
    # Check storage health
    if not verify_storage_health():
        logger.error("ERROR: Storage health check failed")
        return False
    
    logger.info("Environment verification passed")
    return True


def setup_rabbitmq() -> Tuple[pika.BlockingConnection, Any]:
    """Setup RabbitMQ connection with retries and proper consumer configuration."""
    max_retries, retry_delay = 10, 5
    
    for attempt in range(max_retries):
        try:
            # Configure connection with proper parameters for multiple consumers
            parameters = pika.URLParameters(settings.rabbitmq_uri)
            parameters.heartbeat = 600  # 10 minutes heartbeat
            parameters.blocked_connection_timeout = 300  # 5 minutes
            
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()
            
            # Declare queue as durable - remove arguments to avoid conflicts with existing queue
            channel.queue_declare(
                queue=settings.rabbitmq_queue, 
                durable=True
            )
            
            # Critical: Set prefetch to 1 to ensure fair distribution among workers
            # This prevents any single worker from hogging multiple messages
            channel.basic_qos(prefetch_count=1, global_qos=True)
            
            logger.info("Connected to RabbitMQ")
            return connection, channel
            
        except pika.exceptions.AMQPConnectionError as e:
            logger.warning(f"RabbitMQ attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    
    raise ConnectionError(f"Failed to connect to RabbitMQ after {max_retries} attempts")


def message_callback(ch: Any, method: Any, properties: Any, body: bytes) -> None:
    """Process a single ingestion job message."""
    delivery_tag = method.delivery_tag
    logger.info(f"📨 Processing job {delivery_tag}")
    
    try:
        result = process_article_job(body)
        
        # Always ACK to avoid reprocessing
        ch.basic_ack(delivery_tag=delivery_tag)
        
        if result and result.success:
            logger.info(f"Job {delivery_tag} completed: {result.url} ({result.chunks_created} chunks)")
        else:
            error_msg = result.error_message if result else "Unknown error"
            logger.warning(f"Job {delivery_tag} failed: {error_msg}")
            
    except Exception as e:
        logger.error(f"ERROR: Job {delivery_tag} crashed: {e}", exc_info=True)
        ch.basic_nack(delivery_tag=delivery_tag, requeue=False)  # Send to dead letter queue


def main() -> None:
    """Run the simplified LangChain-powered RabbitMQ consumer."""
    logger.info("Starting LangChain ingestion worker...")
    
    load_dotenv()
    
    # Start health check server
    start_health_server(port=8000)
    health_status.update_worker_status(True)
    
    # Verify environment
    if not verify_environment():
        logger.error("💥 Environment verification failed")
        health_status.update_worker_status(False)
        exit(1)
    
    try:
        # Setup RabbitMQ
        connection, channel = setup_rabbitmq()
        
        # Start consuming messages
        channel.basic_consume(queue=settings.rabbitmq_queue, on_message_callback=message_callback)

        logger.info(f"Worker ready - waiting for messages (Queue: {settings.rabbitmq_queue})")
        logger.info(f"Analysis mode: {settings.analysis_mode}")
        logger.info(f"ChromaDB: {settings.chroma_host}")
        
        try:
            channel.start_consuming()
        except KeyboardInterrupt:
            logger.info("Stopping worker...")
            health_status.update_worker_status(False)
            channel.stop_consuming()
            connection.close()
            logger.info("Worker stopped gracefully")
            
    except Exception as e:
        logger.error(f"FATAL: {e}", exc_info=True)
        health_status.update_worker_status(False)
        exit(1)


if __name__ == '__main__':
    main()