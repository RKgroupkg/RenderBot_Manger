{
  "webapp": {
    "health_endpoint": "https://myapp.example.com/health",
    "ping_interval": 60,
    "retry_count": 3,
    "retry_delay": 5,
    "timeout": 10,
    "deploy_endpoint": "https://deploy.example.com/trigger/webapp",
    "deploy_cooldown": 300,
    "headers": {
      "X-Api-Key": "abc123",
      "Accept": "application/json"
    },
    "expected_status": 200,
    "expected_response": {"status": "healthy"}
  },
  "api_service": {
    "health_endpoint": "https://api.example.com/v1/status",
    "ping_interval": 30,
    "retry_count": 5,
    "retry_delay": 3,
    "timeout": 5,
    "deploy_endpoint": "https://deploy.example.com/trigger/api",
    "deploy_cooldown": 180,
    "headers": {
      "Authorization": "Bearer token123"
    },
    "expected_status": 200,
    "expected_response": "regex:\"status\":\\s*\"ok\""
  },
  "database": {
    "health_endpoint": "https://db.example.com/health",
    "ping_interval": 120,
    "retry_count": 3,
    "retry_delay": 10,
    "timeout": 15,
    "deploy_endpoint": "https://deploy.example.com/trigger/database",
    "deploy_cooldown": 600,
    "expected_status": 200,
    "expected_response": {"$.database.status": "online"}
  }
}