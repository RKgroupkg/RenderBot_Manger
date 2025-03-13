"""
Service Monitoring System
-------------------------
A robust service monitoring system that periodically checks the health of multiple web services
and automatically triggers recovery procedures when failures are detected.
"""

import asyncio
import datetime
import enum
import json
import logging
import random
import re
import time
from typing import Any, Dict, List, Optional, Pattern, Set, Tuple, Union
import uuid

import aiohttp
import jsonpath_ng

from keep_alive_ping import KeepAliveService
# For hosting in koyeb and render to keep it alive 
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("service_monitor.log"),
    ]
)
logger = logging.getLogger("service_monitor")
# Setup keepAlive
service = KeepAliveService(
    ping_interval=60,  # Ping every  1 minutes
    log_level=logging.INFO
)

class ServiceState(enum.Enum):
    """Possible states of a monitored service."""
    UP = "UP"                     # Service is healthy
    DEGRADED = "DEGRADED"         # Service is responding but with issues
    DOWN = "DOWN"                 # Service is not responding
    RECOVERING = "RECOVERING"     # Recovery has been triggered
    COOLDOWN = "COOLDOWN"         # In post-deployment cooldown period


class CircuitState(enum.Enum):
    """Circuit breaker states for deployment protection."""
    CLOSED = "CLOSED"       # Normal operation, deployments allowed
    OPEN = "OPEN"           # Too many failures, deployments blocked
    HALF_OPEN = "HALF_OPEN" # Testing if deployments can resume


class ServiceMonitor:
    """
    Monitors the health of web services and triggers automated recovery.
    
    This class handles the monitoring of multiple services concurrently,
    tracks their health state, and initiates recovery procedures when needed.
    """
    
    def __init__(self, config: Dict[str, Dict[str, Any]]) -> None:
        """
        Initialize the service monitor with configuration.
        
        Args:
            config: Dictionary of service configurations
        """
        self.config = config
        self.service_states: Dict[str, ServiceState] = {}
        self.service_metrics: Dict[str, Dict[str, Any]] = {}
        self.last_check_times: Dict[str, float] = {}
        self.last_deploy_times: Dict[str, float] = {}
        self.failed_attempts: Dict[str, int] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Circuit breaker settings
        self.circuit_state = CircuitState.CLOSED
        self.circuit_failure_threshold = 5
        self.circuit_timeout = 300  # seconds
        self.circuit_last_failure_time = 0
        self.circuit_failure_count = 0
        
        # Initialize service states and metrics
        for service_name in config:
            self.service_states[service_name] = ServiceState.UP
            self.service_metrics[service_name] = {
                "total_checks": 0,
                "failures": 0,
                "recoveries": 0,
                "last_response_time": None,
                "response_times": [],  # recent response times for trending
                "uptime_periods": [],  # list of (start_time, end_time) for UP periods
                "downtime_periods": [],  # list of (start_time, end_time) for DOWN periods
                "last_status_change": time.time(),
                "last_error": None,
            }
            self.last_check_times[service_name] = 0
            self.last_deploy_times[service_name] = 0
            self.failed_attempts[service_name] = 0
    
    async def start(self) -> None:
        """Start the monitoring process."""
        logger.info("Starting service monitor")
        self.session = aiohttp.ClientSession()
        try:
            await self._monitor_loop()
        finally:
            await self.session.close()
            self.session = None
            logger.info("Service monitor stopped")
    
    async def stop(self) -> None:
        """Stop the monitoring process."""
        # This will be implemented to gracefully stop the monitor
        # For now, we're using KeyboardInterrupt to stop it
        pass
    
    async def _monitor_loop(self) -> None:
        """Main monitoring loop that periodically checks all services."""
        while True:
            check_tasks = []
            current_time = time.time()
            
            for service_name, service_config in self.config.items():
                # Skip services in cooldown
                if self._is_in_cooldown(service_name):
                    continue
                
                # Check if it's time to check this service
                ping_interval = service_config["ping_interval"]
                if current_time - self.last_check_times.get(service_name, 0) >= ping_interval:
                    # Add jitter to prevent thundering herd
                    jitter = random.uniform(0, 0.1 * ping_interval)
                    await asyncio.sleep(jitter)
                    
                    # Schedule the health check
                    task = asyncio.create_task(
                        self._check_service_health(service_name, service_config)
                    )
                    check_tasks.append(task)
                    self.last_check_times[service_name] = current_time
            
            # Wait for all scheduled checks to complete
            if check_tasks:
                await asyncio.gather(*check_tasks, return_exceptions=True)
            
            # Brief sleep to prevent busy waiting
            await asyncio.sleep(1)
    
    async def _check_service_health(
        self, service_name: str, service_config: Dict[str, Any]
    ) -> None:
        """
        Check the health of a specific service.
        
        Args:
            service_name: Name of the service to check
            service_config: Configuration for the service
        """
        logger.debug(f"Checking health of {service_name}")
        
        health_endpoint = service_config["health_endpoint"]
        timeout = service_config["timeout"]
        headers = service_config.get("headers", {})
        expected_status = service_config["expected_status"]
        expected_response = service_config.get("expected_response")
        
        start_time = time.time()
        is_healthy = False
        error_details = None
        response_body = None
        
        try:
            if not self.session:
                raise RuntimeError("HTTP session not initialized")
            
            # Perform the health check request
            async with self.session.get(
                health_endpoint,
                timeout=aiohttp.ClientTimeout(total=timeout),
                headers=headers,
                ssl=True,
            ) as response:
                response_body = await response.text()
                response_time = time.time() - start_time
                
                # Update response time metrics
                metrics = self.service_metrics[service_name]
                metrics["last_response_time"] = response_time
                metrics["response_times"].append(response_time)
                # Keep only recent response times for trending
                if len(metrics["response_times"]) > 100:
                    metrics["response_times"].pop(0)
                
                # Check if status code matches expected
                if response.status == expected_status:
                    # Check response content if expected pattern is defined
                    if expected_response is not None:
                        is_healthy = self._validate_response(
                            response_body, expected_response
                        )
                        if not is_healthy:
                            error_details = "Response validation failed"
                    else:
                        is_healthy = True
                else:
                    error_details = f"Unexpected status code: {response.status}"
                    
        except asyncio.TimeoutError:
            error_details = f"Request timed out after {timeout}s"
        except aiohttp.ClientError as e:
            error_details = f"Connection error: {str(e)}"
        except Exception as e:
            error_details = f"Unexpected error: {str(e)}"
        
        # Update metrics
        self.service_metrics[service_name]["total_checks"] += 1
        
        # Process the health check result
        await self._process_health_result(
            service_name, is_healthy, error_details, response_body
        )
    
    def _validate_response(
        self, response_body: str, expected_pattern: Union[str, Dict[str, Any]]
    ) -> bool:
        """
        Validate the response body against an expected pattern.
        
        Args:
            response_body: The response body to validate
            expected_pattern: String, regex pattern, or JSON path to validate against
            
        Returns:
            True if validation passes, False otherwise
        """
        # If expected_pattern is a dict, treat it as a JSON path validation
        if isinstance(expected_pattern, dict):
            try:
                resp_json = json.loads(response_body)
                for json_path, expected_value in expected_pattern.items():
                    # Use jsonpath-ng to extract values
                    jsonpath_expr = jsonpath_ng.parse(json_path)
                    matches = [match.value for match in jsonpath_expr.find(resp_json)]
                    
                    if not matches or expected_value not in matches:
                        return False
                return True
            except (json.JSONDecodeError, Exception):
                return False
        
        # If expected_pattern starts with 'regex:', treat it as a regex pattern
        elif isinstance(expected_pattern, str) and expected_pattern.startswith("regex:"):
            pattern = expected_pattern[6:]  # Remove 'regex:' prefix
            try:
                return bool(re.search(pattern, response_body))
            except re.error:
                logger.error(f"Invalid regex pattern: {pattern}")
                return False
        
        # Otherwise, treat it as a simple string match
        else:
            return expected_pattern in response_body
    
    async def _process_health_result(
        self,
        service_name: str,
        is_healthy: bool,
        error_details: Optional[str],
        response_body: Optional[str],
    ) -> None:
        """
        Process the result of a health check and update service state.
        
        Args:
            service_name: Name of the service
            is_healthy: Whether the service is healthy
            error_details: Details of any error that occurred
            response_body: The response body from the health check
        """
        current_state = self.service_states[service_name]
        metrics = self.service_metrics[service_name]
        service_config = self.config[service_name]
        retry_count = service_config["retry_count"]
        retry_delay = service_config["retry_delay"]
        
        if is_healthy:
            # Service is healthy
            if current_state != ServiceState.UP:
                # Transition to UP state
                old_state = current_state
                self.service_states[service_name] = ServiceState.UP
                
                # Log state transition
                logger.info(
                    f"Service {service_name} transitioned from {old_state.value} to UP"
                )
                
                # Update metrics
                if old_state == ServiceState.DOWN or old_state == ServiceState.DEGRADED:
                    # End any active downtime period
                    if metrics["downtime_periods"] and len(metrics["downtime_periods"][-1]) == 1:
                        metrics["downtime_periods"][-1] = (
                            metrics["downtime_periods"][-1][0],
                            time.time(),
                        )
                    
                    # Start a new uptime period
                    metrics["uptime_periods"].append((time.time(),))
                
                metrics["last_status_change"] = time.time()
            
            # Reset failed attempt counter
            self.failed_attempts[service_name] = 0
            
        else:
            # Service is not healthy
            metrics["last_error"] = error_details
            
            # Increment failed attempts
            self.failed_attempts[service_name] += 1
            current_attempts = self.failed_attempts[service_name]
            
            if current_attempts < retry_count:
                # Still within retry limits
                if current_state == ServiceState.UP:
                    # Transition to DEGRADED state
                    logger.warning(
                        f"Service {service_name} is degraded: {error_details}. "
                        f"Attempt {current_attempts}/{retry_count}"
                    )
                    self.service_states[service_name] = ServiceState.DEGRADED
                    metrics["last_status_change"] = time.time()
                
                # Schedule a retry after delay
                retry_with_backoff = retry_delay * (2 ** (current_attempts - 1))
                logger.info(
                    f"Scheduling retry for {service_name} in {retry_with_backoff} seconds"
                )
                # Adjust the last check time to trigger a quicker re-check
                self.last_check_times[service_name] = (
                    time.time() - service_config["ping_interval"] + retry_with_backoff
                )
                
            else:
                # Exceeded retry count, mark as DOWN
                metrics["failures"] += 1
                
                if current_state != ServiceState.DOWN:
                    logger.error(
                        f"Service {service_name} is DOWN after {retry_count} failed attempts. "
                        f"Last error: {error_details}"
                    )
                    self.service_states[service_name] = ServiceState.DOWN
                    
                    # Start a new downtime period if needed
                    if not metrics["downtime_periods"] or len(metrics["downtime_periods"][-1]) == 2:
                        metrics["downtime_periods"].append((time.time(),))
                    
                    # End any active uptime period
                    if metrics["uptime_periods"] and len(metrics["uptime_periods"][-1]) == 1:
                        metrics["uptime_periods"][-1] = (
                            metrics["uptime_periods"][-1][0],
                            time.time(),
                        )
                    
                    metrics["last_status_change"] = time.time()
                    
                    # Trigger recovery process
                    await self._trigger_recovery(service_name)
                else:
                    logger.warning(
                        f"Service {service_name} remains DOWN. "
                        f"Last error: {error_details}"
                    )
    
    async def _trigger_recovery(self, service_name: str) -> None:
        """
        Trigger recovery for a service by calling its deployment endpoint.
        
        Args:
            service_name: Name of the service to recover
        """
        # Check if circuit breaker allows deployment
        if not self._circuit_allows_deployment():
            logger.warning(
                f"Circuit breaker is OPEN - skipping recovery for {service_name}"
            )
            return
        
        service_config = self.config[service_name]
        deploy_endpoint = service_config["deploy_endpoint"]
        deploy_cooldown = service_config["deploy_cooldown"]
        
        logger.info(f"Triggering recovery for service {service_name}")
        self.service_states[service_name] = ServiceState.RECOVERING
        
        success = False
        try:
            if not self.session:
                raise RuntimeError("HTTP session not initialized")
            
            # Call deployment endpoint
            async with self.session.post(
                deploy_endpoint,
                timeout=aiohttp.ClientTimeout(total=60),
                headers=service_config.get("deploy_headers", {}),
            ) as response:
                if response.status in (200, 201, 202, 204):
                    logger.info(
                        f"Successfully triggered deployment for {service_name}. "
                        f"Entering cooldown for {deploy_cooldown} seconds"
                    )
                    success = True
                else:
                    response_text = await response.text()
                    logger.error(
                        f"Failed to trigger deployment for {service_name}. "
                        f"Status: {response.status}, Response: {response_text}"
                    )
        except Exception as e:
            logger.error(f"Error triggering deployment for {service_name}: {str(e)}")
            
        # Update circuit breaker state
        self._update_circuit_state(success)
            
        # Reset retry counter
        self.failed_attempts[service_name] = 0
        
        # Update metrics and enter cooldown state
        self.last_deploy_times[service_name] = time.time()
        self.service_states[service_name] = ServiceState.COOLDOWN
        self.service_metrics[service_name]["recoveries"] += 1
        
        # Schedule a task to exit cooldown after the specified period
        asyncio.create_task(self._exit_cooldown(service_name, deploy_cooldown))
    
    async def _exit_cooldown(self, service_name: str, cooldown_period: int) -> None:
        """
        Schedule service to exit cooldown state after the cooldown period.
        
        Args:
            service_name: Name of the service
            cooldown_period: Cooldown period in seconds
        """
        await asyncio.sleep(cooldown_period)
        
        if self.service_states[service_name] == ServiceState.COOLDOWN:
            logger.info(f"Service {service_name} exiting cooldown state")
            self.service_states[service_name] = ServiceState.UP
            
            # Force an immediate health check
            self.last_check_times[service_name] = 0
    
    def _is_in_cooldown(self, service_name: str) -> bool:
        """
        Check if a service is currently in cooldown.
        
        Args:
            service_name: Name of the service
            
        Returns:
            True if the service is in cooldown, False otherwise
        """
        if self.service_states[service_name] != ServiceState.COOLDOWN:
            return False
            
        service_config = self.config[service_name]
        deploy_cooldown = service_config["deploy_cooldown"]
        time_since_deploy = time.time() - self.last_deploy_times.get(service_name, 0)
        
        return time_since_deploy < deploy_cooldown
    
    def _circuit_allows_deployment(self) -> bool:
        """
        Check if the circuit breaker allows deployment.
        
        Returns:
            True if deployment is allowed, False otherwise
        """
        if self.circuit_state == CircuitState.CLOSED:
            return True
            
        if self.circuit_state == CircuitState.OPEN:
            # Check if it's time to transition to HALF_OPEN
            if time.time() - self.circuit_last_failure_time > self.circuit_timeout:
                logger.info("Circuit breaker transitioning from OPEN to HALF_OPEN")
                self.circuit_state = CircuitState.HALF_OPEN
                return True
            return False
            
        # In HALF_OPEN state, allow one deployment to test the waters
        return True
    
    def _update_circuit_state(self, deployment_success: bool) -> None:
        """
        Update circuit breaker state based on deployment result.
        
        Args:
            deployment_success: Whether the deployment was successful
        """
        if deployment_success:
            if self.circuit_state == CircuitState.HALF_OPEN:
                logger.info("Circuit breaker transitioning from HALF_OPEN to CLOSED")
                self.circuit_state = CircuitState.CLOSED
                self.circuit_failure_count = 0
        else:
            self.circuit_failure_count += 1
            self.circuit_last_failure_time = time.time()
            
            if (self.circuit_state == CircuitState.CLOSED and 
                    self.circuit_failure_count >= self.circuit_failure_threshold):
                logger.warning(
                    f"Circuit breaker transitioning from CLOSED to OPEN after "
                    f"{self.circuit_failure_count} consecutive failures"
                )
                self.circuit_state = CircuitState.OPEN
            elif self.circuit_state == CircuitState.HALF_OPEN:
                logger.warning("Circuit breaker transitioning from HALF_OPEN back to OPEN")
                self.circuit_state = CircuitState.OPEN
    
    def get_service_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the current status of all monitored services.
        
        Returns:
            Dictionary with service status information
        """
        result = {}
        for service_name in self.config:
            metrics = self.service_metrics[service_name]
            cooldown_remaining = 0
            
            if self._is_in_cooldown(service_name):
                service_config = self.config[service_name]
                cooldown_total = service_config["deploy_cooldown"]
                time_elapsed = time.time() - self.last_deploy_times.get(service_name, 0)
                cooldown_remaining = max(0, cooldown_total - time_elapsed)
            
            # Calculate uptime percentage
            uptime_seconds = 0
            downtime_seconds = 0
            
            for period in metrics["uptime_periods"]:
                if len(period) == 1:  # Still ongoing
                    uptime_seconds += time.time() - period[0]
                else:  # Completed period
                    uptime_seconds += period[1] - period[0]
                    
            for period in metrics["downtime_periods"]:
                if len(period) == 1:  # Still ongoing
                    downtime_seconds += time.time() - period[0]
                else:  # Completed period
                    downtime_seconds += period[1] - period[0]
            
            total_tracked_time = uptime_seconds + downtime_seconds
            uptime_percentage = 0
            if total_tracked_time > 0:
                uptime_percentage = (uptime_seconds / total_tracked_time) * 100
            
            # Calculate average response time
            response_times = metrics["response_times"]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            
            result[service_name] = {
                "state": self.service_states[service_name].value,
                "last_check": self.last_check_times.get(service_name, 0),
                "last_status_change": metrics["last_status_change"],
                "last_deploy": self.last_deploy_times.get(service_name, 0),
                "cooldown_remaining": cooldown_remaining,
                "uptime_percentage": round(uptime_percentage, 2),
                "failure_count": metrics["failures"],
                "recovery_count": metrics["recoveries"],
                "last_error": metrics["last_error"],
                "last_response_time": metrics["last_response_time"],
                "avg_response_time": avg_response_time,
            }
        
        return result


class ServiceMonitorCLI:
    """Command-line interface for the service monitor."""
    
    def __init__(self, config_path: str) -> None:
        """
        Initialize the CLI.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.monitor = None
    
    def load_config(self) -> Dict[str, Dict[str, Any]]:
        """
        Load configuration from file.
        
        Returns:
            Service configuration dictionary
        """
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    async def start(self) -> None:
        """Start the monitoring service."""
        config = self.load_config()
        self.monitor = ServiceMonitor(config)
        await self.monitor.start()
    
    async def display_status(self) -> None:
        """Display the current status of all services."""
        if not self.monitor:
            print("Monitor not started")
            return
            
        status = self.monitor.get_service_status()
        
        # Format and display the status
        print("\n" + "=" * 80)
        print(f"SERVICE MONITOR STATUS - {datetime.datetime.now().isoformat()}")
        print("=" * 80)
        
        for service_name, service_status in status.items():
            state = service_status["state"]
            uptime = service_status["uptime_percentage"]
            last_check = datetime.datetime.fromtimestamp(service_status["last_check"]).isoformat()
            
            # Use ANSI colors for different states
            color_code = {
                "UP": "\033[92m",  # Green
                "DEGRADED": "\033[93m",  # Yellow
                "DOWN": "\033[91m",  # Red
                "RECOVERING": "\033[94m",  # Blue
                "COOLDOWN": "\033[96m",  # Cyan
            }.get(state, "\033[0m")
            
            reset_code = "\033[0m"
            
            print(f"{service_name}: {color_code}{state}{reset_code}")
            print(f"  Uptime: {uptime}%")
            print(f"  Last Check: {last_check}")
            
            if state == "COOLDOWN":
                cooldown = service_status["cooldown_remaining"]
                print(f"  Cooldown: {int(cooldown)} seconds remaining")
                
            if service_status["last_error"]:
                print(f"  Last Error: {service_status['last_error']}")
                
            print("")
        
        print("=" * 80)
        return status

async def main() -> None:
    """Main entry point for the service monitor CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Service Monitoring System")
    parser.add_argument(
        "--config", 
        default="config.json", 
        help="Path to configuration file"
    )
    parser.add_argument(
        "--status-interval", 
        type=int, 
        default=60, 
        help="Status display interval in seconds"
    )
    args = parser.parse_args()
    
    cli = ServiceMonitorCLI(args.config)
    bot = TelegramServiceStatusBot(os.environ.get('Telegram_token'))
    
    # Start the monitor in a separate task
    monitor_task = asyncio.create_task(cli.start())
    
    try:
        # Display status periodically
        while True:
            await bot.edit_service_status_message(
        chat_id=1002399739583,
        message_id=10,
        services_data= await cli.display_status()
    )

            
            await asyncio.sleep(args.status_interval)
    except KeyboardInterrupt:
        logger.info("Shutting down monitor")
    finally:
        # Cancel the monitor task
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    service.start() # Start my api monitor
    asyncio.run(main())