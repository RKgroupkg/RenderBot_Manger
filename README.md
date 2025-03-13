# Service Monitoring System

A robust service monitoring system that periodically checks the health of multiple web services and automatically triggers recovery procedures when failures are detected.

## Features

- **Concurrent Monitoring**: Monitors multiple services in parallel with customizable checks
- **Automatic Recovery**: Triggers deployment endpoints when services fail
- **Circuit Breaker**: Prevents cascading failures during recovery attempts
- **Cooldown Management**: Allows services time to stabilize after recovery
- **Detailed Metrics**: Tracks uptime, response times, and failure history

## Installation

```bash
git clone <repository-url>
cd service-monitor
pip install -r requirements.txt
```

## Configuration

Create a `config.json` file with your service definitions:

```json
{
  "service-name": {
    "health_endpoint": "https://example.com/health",
    "deploy_endpoint": "https://example.com/deploy",
    "expected_status": 200,
    "expected_response": {"status": "healthy"},
    "ping_interval": 60,
    "timeout": 10,
    "retry_count": 3,
    "retry_delay": 5,
    "deploy_cooldown": 300,
    "headers": {
      "Authorization": "Bearer token"
    }
  }
}
```

## Usage

```bash
# Start monitoring with default config.json
python service_monitor.py

# Use custom config and status display interval
python service_monitor.py --config my_config.json --status-interval 30
```

## Service States

- **UP**: Service is healthy
- **DEGRADED**: Service is responding but with issues
- **DOWN**: Service is not responding
- **RECOVERING**: Recovery has been triggered
- **COOLDOWN**: In post-deployment cooldown period

## Note

This system is not intended for public use. It was developed specifically for monitoring internal bot services.