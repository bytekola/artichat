"""Ingestion service - queues jobs for worker and checks status."""

import json

import aio_pika
import redis.asyncio as redis
from app.core.config import settings


class IngestionService:
    """Service to queue ingestion jobs and check status."""
    
    def __init__(self):
        self.rabbitmq = None
        self.redis = None
    
    async def connect(self):
        """Connect to RabbitMQ and Redis."""
        # Connect to RabbitMQ for job queuing
        self.rabbitmq = await aio_pika.connect_robust(settings.rabbitmq_url)
        channel = await self.rabbitmq.channel()
        
        # Declare queue (create if not exists)
        await channel.declare_queue(settings.rabbitmq_queue, durable=True)
        
        # Connect to Redis for status checking
        self.redis = redis.from_url(settings.redis_url)

    async def queue_job(self, url: str, overwrite: bool = False) -> dict:
        """Queue an ingestion job."""
        if not self.rabbitmq:
            raise RuntimeError("RabbitMQ connection not established. Call connect() first.")
            
        job_data = {
            "url": url,
            "priority": 5,
            "retry_count": 0,
            "created_at": "2025-09-01T12:00:00Z" 
        }
        if overwrite:
            job_data["overwrite"] = True

        channel = await self.rabbitmq.channel()
        message = aio_pika.Message(
            json.dumps(job_data).encode(),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT
        )
        
        await channel.default_exchange.publish(
            message,
            routing_key=settings.rabbitmq_queue
        )
        
        return {
            "message": "Article queued for processing",
            "url": url,
            "status": "accepted"
        }
    
    async def get_status(self, url: str) -> dict:
        """Get processing status from Redis."""
        if not self.redis:
            return {
                "url": url,
                "status": "error",
                "chunks_created": 0,
                "error_message": "Redis connection not established"
            }
            
        status_data = await self.redis.get(f"ingestion:{url}")
        
        if not status_data:
            return {
                "url": url,
                "status": "not_found",
                "chunks_created": 0,
                "error_message": "No processing record found"
            }
        
        data = json.loads(status_data)
        return {
            "url": url,
            "status": data.get("status", "unknown"),
            "chunks_created": data.get("chunks_created", 0),
            "error_message": data.get("error_message")
        }
    
    async def close(self):
        """Close connections."""
        if self.rabbitmq:
            await self.rabbitmq.close()
        if self.redis:
            await self.redis.close()
