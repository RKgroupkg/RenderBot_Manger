import asyncio
import time
from enum import Enum
from datetime import datetime
from typing import Dict, Any, Optional
import logging

# Telegram API library
import telegram
from telegram.ext import Application

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

class ServiceState(Enum):
    RUNNING = "RUNNING"
    DEGRADED = "DEGRADED"
    DOWN = "DOWN"
    MAINTENANCE = "MAINTENANCE"
    UNKNOWN = "UNKNOWN"

class TelegramServiceStatusBot:
    def __init__(self, token: str):
        """
        Initialize the Telegram bot for service status updates.
        
        Args:
            token (str): Telegram Bot API token
        """
        self.token = token
        self.bot = Application.builder().token(token).build()
        
    async def edit_service_status_message(self, chat_id: int, message_id: int, services_data: Dict[str, Dict[str, Any]]):
        """
        Edit an existing message with formatted service status information.
        
        Args:
            chat_id (int): Telegram chat ID
            message_id (int): ID of the message to edit
            services_data (Dict[str, Dict]): Dictionary of service names to their status data
        """
        formatted_message = self._format_services_status(services_data)
        
        try:
            await self.bot.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=formatted_message,
                parse_mode="Markdown",
                disable_web_page_preview=True
            )
            logger.info(f"Successfully updated status message (chat_id: {chat_id}, message_id: {message_id})")
        except Exception as e:
            logger.error(f"Failed to edit message: {e}")
            
    def _format_services_status(self, services_data: Dict[str, Dict[str, Any]]) -> str:
        """
        Format multiple services status data into a single message.
        
        Args:
            services_data (Dict[str, Dict]): Dictionary of service names to their status data
            
        Returns:
            str: Formatted status message for all services
        """
        if not services_data:
            return "No service data available."
            
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message_parts = [f"*Service Status Dashboard* (as of {current_time})\n"]
        
        # Sort services by state (DOWN first, then DEGRADED, etc.)
        def get_state_priority(service_name):
            state = services_data[service_name].get("state", ServiceState.UNKNOWN.value)
            priorities = {
                ServiceState.DOWN.value: 0,
                ServiceState.DEGRADED.value: 1,
                ServiceState.MAINTENANCE.value: 2,
                ServiceState.RUNNING.value: 3,
                ServiceState.UNKNOWN.value: 4
            }
            return priorities.get(state, 5)
            
        sorted_services = sorted(services_data.keys(), key=get_state_priority)
        
        for service_name in sorted_services:
            service_data = services_data[service_name]
            message_parts.append(self._format_single_service(service_name, service_data))
            
        return "\n\n".join(message_parts)
    
    def _format_single_service(self, service_name: str, service_data: Dict[str, Any]) -> str:
        """
        Format a single service's status data.
        
        Args:
            service_name (str): Name of the service
            service_data (dict): Dictionary containing service metrics
            
        Returns:
            str: Formatted status for a single service
        """
        state = service_data.get("state", ServiceState.UNKNOWN.value)
        last_check = service_data.get("last_check", 0)
        last_status_change = service_data.get("last_status_change", 0)
        uptime_percentage = service_data.get("uptime_percentage", 0)
        failure_count = service_data.get("failure_count", 0)
        last_response_time = service_data.get("last_response_time", 0)
        avg_response_time = service_data.get("avg_response_time", 0)
        last_error = service_data.get("last_error", "")
        
        # Get status icon
        icon = self._get_state_icon(state)
        
        # Format time since last status change
        time_since_change = self._format_duration(time.time() - last_status_change) if last_status_change else "N/A"
        
        # Build the service status section
        parts = [
            f"{icon} *{service_name}*: `{state}`",
            f"• Uptime: `{uptime_percentage}%`",
            f"• Response: `{last_response_time}ms` (avg: `{avg_response_time}ms`)",
            f"• Status duration: `{time_since_change}`",
            f"• Failures: `{failure_count}`"
        ]
        
        # Add last error if present
        if last_error:
            # Truncate very long error messages
            if len(last_error) > 100:
                last_error = last_error[:97] + "..."
            parts.append(f"• Last error: `{last_error}`")
            
        # Add last check time
        if last_check:
            check_time = datetime.fromtimestamp(last_check).strftime("%H:%M:%S")
            parts.append(f"• Last checked: `{check_time}`")
            
        return "\n".join(parts)
    
    def _get_state_icon(self, state: str) -> str:
        """Get an appropriate icon for the service state."""
        if state == ServiceState.RUNNING.value:
            return "✅"  # Green checkmark
        elif state == ServiceState.DEGRADED.value:
            return "⚠️"  # Warning
        elif state == ServiceState.DOWN.value:
            return "❌"  # Red X
        elif state == ServiceState.MAINTENANCE.value:
            return "🔧"  # Wrench
        else:
            return "❓"  # Question mark for unknown state
    
    def _format_timestamp(self, timestamp: float) -> str:
        """Format a timestamp into a human-readable datetime."""
        if not timestamp:
            return "Never"
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    
    def _format_duration(self, seconds: float) -> str:
        """Format seconds into a human-readable duration."""
        if not seconds or seconds <= 0:
            return "0s"
        
        minutes, seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        days, hours = divmod(hours, 24)
        
        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if seconds > 0:
            parts.append(f"{seconds}s")
        
        return " ".join(parts[:2])  # Show at most 2 units for brevity

# Example usage
async def example():
    # Replace with your actual Telegram bot token
    bot = TelegramServiceStatusBot("YOUR_TELEGRAM_BOT_TOKEN")
    
    # Example service data
    services_data = {
        "api-gateway": {
            "state": "RUNNING",
            "last_check": time.time() - 120,  # 2 minutes ago
            "last_status_change": time.time() - 86400,  # 1 day ago
            "last_deploy": time.time() - 604800,  # 1 week ago
            "cooldown_remaining": 0,
            "uptime_percentage": 99.95,
            "failure_count": 2,
            "recovery_count": 2,
            "last_error": "",
            "last_response_time": 350.5,
            "avg_response_time": 275.3,
        },
        "user-service": {
            "state": "DEGRADED",
            "last_check": time.time() - 60,  # 1 minute ago
            "last_status_change": time.time() - 3600,  # 1 hour ago
            "last_deploy": time.time() - 86400,  # 1 day ago
            "cooldown_remaining": 300,  # 5 minutes
            "uptime_percentage": 95.75,
            "failure_count": 5,
            "recovery_count": 4,
            "last_error": "High CPU load: 92%",
            "last_response_time": 1250.8,
            "avg_response_time": 450.2,
        },
        "database": {
            "state": "DOWN",
            "last_check": time.time() - 30,  # 30 seconds ago
            "last_status_change": time.time() - 300,  # 5 minutes ago
            "last_deploy": time.time() - 259200,  # 3 days ago
            "cooldown_remaining": 0,
            "uptime_percentage": 99.5,
            "failure_count": 1,
            "recovery_count": 0,
            "last_error": "Connection refused: timeout after 5s",
            "last_response_time": 0,
            "avg_response_time": 120.5,
        }
    }
    
    # Replace with actual chat_id and message_id
    await bot.edit_service_status_message(
        chat_id=123456789,
        message_id=987654321,
        services_data=services_data
    )

if __name__ == "__main__":
    # For testing the example
    # asyncio.run(example())
    pass