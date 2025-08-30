# 🚀 Deployment Guide & Configuration

## Advanced Stock Predictor AI - Production Deployment

### 🎯 **Deployment Overview**

This guide covers multiple deployment strategies for the Advanced Stock Predictor AI, from local development to enterprise-grade production environments.

---

## 📋 **Pre-Deployment Checklist**

### **✅ System Requirements**

#### **Minimum Hardware Requirements**
```
CPU: 2 cores, 2.4GHz
RAM: 4GB minimum, 8GB recommended
Storage: 10GB free space
Network: Stable internet connection (10+ Mbps)
```

#### **Recommended Hardware Requirements**
```
CPU: 4+ cores, 3.0GHz+
RAM: 16GB+ for optimal performance
Storage: 50GB+ SSD storage
Network: High-speed internet (50+ Mbps)
GPU: Optional, for advanced ML models
```

#### **Software Prerequisites**
```
Operating System: Ubuntu 20.04+, Windows 10+, macOS 11+
Python: 3.10+ (Recommended: 3.12.4)
Docker: 20.10+ (for containerized deployment)
Git: 2.30+
Node.js: 16+ (for advanced build tools)
```

### **🔐 Security Checklist**

- [ ] **Environment Variables:** All sensitive data in environment variables
- [ ] **SSL/TLS:** HTTPS enabled for production
- [ ] **Authentication:** Strong session management implemented
- [ ] **Rate Limiting:** API rate limits configured
- [ ] **Input Validation:** All user inputs validated and sanitized
- [ ] **CORS:** Cross-origin requests properly configured
- [ ] **Headers:** Security headers implemented
- [ ] **Logging:** Comprehensive audit logging enabled

### **📊 Performance Checklist**

- [ ] **Caching:** Multi-level caching strategy implemented
- [ ] **Database:** Optimized database queries and indexing
- [ ] **CDN:** Static assets served via CDN
- [ ] **Compression:** Gzip/Brotli compression enabled
- [ ] **Monitoring:** Performance monitoring tools configured
- [ ] **Load Testing:** Application tested under load
- [ ] **Scaling:** Auto-scaling policies defined

---

## 🖥️ **Local Development Deployment**

### **Quick Start (5 Minutes)**

```bash
# 1. Clone repository
git clone https://github.com/your-username/advanced-stock-predictor.git
cd advanced-stock-predictor

# 2. Create virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run application
streamlit run main.py

# 5. Open browser
# http://localhost:8501
```

### **Development Configuration**

#### **Environment Setup**

**`.env.development`:**
```bash
# Development environment
ENVIRONMENT=development
DEBUG=True
LOG_LEVEL=DEBUG

# Application settings
STREAMLIT_PORT=8501
STREAMLIT_HOST=localhost
CACHE_TTL=300  # 5 minutes for development

# API settings
YAHOO_FINANCE_TIMEOUT=30
MAX_REQUESTS_PER_MINUTE=100

# Security (development only)
SECRET_KEY=dev-secret-key-change-in-production
SESSION_TIMEOUT=7200  # 2 hours

# Database (SQLite for development)
DATABASE_URL=sqlite:///./dev_database.db

# External services
REDIS_URL=redis://localhost:6379/0
SENTRY_DSN=  # Empty for development
```

#### **Development Tools Setup**

**VS Code Extensions:**
```json
{
    "recommendations": [
        "ms-python.python",
        "ms-python.black-formatter",
        "ms-python.pylint",
        "ms-python.isort",
        "ms-toolsai.jupyter",
        "ms-vscode.vscode-typescript-next",
        "bradlc.vscode-tailwindcss",
        "formulahendry.auto-rename-tag"
    ]
}
```

**Pre-commit Hooks (`pyproject.toml`):**
```toml
[tool.black]
line-length = 88
target-version = ['py310']

[tool.isort]
profile = "black"
multi_line_output = 3

[tool.pylint.messages_control]
disable = "C0103,R0903,R0913"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
```

---

## 🐳 **Docker Deployment**

### **Single Container Deployment**

#### **Production Dockerfile**

```dockerfile
# Multi-stage build for optimization
FROM python:3.12-slim as builder

# Set build arguments
ARG ENVIRONMENT=production
ARG BUILD_DATE
ARG VERSION

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Production stage
FROM python:3.12-slim as production

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create app user
RUN groupadd -g 1000 appgroup && \
    useradd -r -u 1000 -g appgroup appuser

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appgroup . .

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs && \
    chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Add labels
LABEL maintainer="your-email@example.com" \
      version="${VERSION}" \
      description="Advanced Stock Predictor AI" \
      build-date="${BUILD_DATE}"

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run application
CMD ["streamlit", "run", "main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
```

#### **Docker Compose Configuration**

**`docker-compose.yml`:**
```yaml
version: '3.8'

services:
  # Main application
  stock-predictor:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        ENVIRONMENT: production
        BUILD_DATE: ${BUILD_DATE:-}
        VERSION: ${VERSION:-latest}
    container_name: stock-predictor-app
    ports:
      - "8501:8501"
    environment:
      - ENVIRONMENT=production
      - REDIS_URL=redis://redis:6379/0
      - DATABASE_URL=postgresql://stockuser:${POSTGRES_PASSWORD}@postgres:5432/stockdb
    env_file:
      - .env.production
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
    depends_on:
      - redis
      - postgres
    restart: unless-stopped
    networks:
      - stock-predictor-network
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'

  # Redis for caching
  redis:
    image: redis:7-alpine
    container_name: stock-predictor-redis
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    restart: unless-stopped
    networks:
      - stock-predictor-network

  # PostgreSQL database
  postgres:
    image: postgres:15-alpine
    container_name: stock-predictor-db
    environment:
      POSTGRES_DB: stockdb
      POSTGRES_USER: stockuser
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped
    networks:
      - stock-predictor-network

  # Nginx reverse proxy
  nginx:
    image: nginx:alpine
    container_name: stock-predictor-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
      - ./nginx/logs:/var/log/nginx
    depends_on:
      - stock-predictor
    restart: unless-stopped
    networks:
      - stock-predictor-network

  # Monitoring with Prometheus
  prometheus:
    image: prom/prometheus:latest
    container_name: stock-predictor-prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    restart: unless-stopped
    networks:
      - stock-predictor-network

  # Grafana for visualization
  grafana:
    image: grafana/grafana:latest
    container_name: stock-predictor-grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
    ports:
      - "3000:3000"
    restart: unless-stopped
    networks:
      - stock-predictor-network

volumes:
  redis_data:
  postgres_data:
  prometheus_data:
  grafana_data:

networks:
  stock-predictor-network:
    driver: bridge
```

#### **Production Environment Variables**

**`.env.production`:**
```bash
# Build information
BUILD_DATE=2024-01-15T10:30:00Z
VERSION=1.0.0

# Application configuration
ENVIRONMENT=production
DEBUG=False
LOG_LEVEL=INFO
SECRET_KEY=your-super-secret-production-key

# Server settings
STREAMLIT_PORT=8501
STREAMLIT_HOST=0.0.0.0
MAX_UPLOAD_SIZE=200MB

# Database credentials
POSTGRES_PASSWORD=your-secure-db-password
DATABASE_URL=postgresql://stockuser:your-secure-db-password@postgres:5432/stockdb

# Redis settings
REDIS_PASSWORD=your-secure-redis-password
REDIS_URL=redis://:your-secure-redis-password@redis:6379/0

# External API settings
YAHOO_FINANCE_TIMEOUT=10
MAX_REQUESTS_PER_MINUTE=60

# Monitoring
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id
GRAFANA_PASSWORD=your-grafana-password

# Security settings
SESSION_TIMEOUT=3600
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
```

### **Deployment Commands**

```bash
# Build and deploy
docker-compose up -d --build

# View logs
docker-compose logs -f stock-predictor

# Scale services
docker-compose up -d --scale stock-predictor=3

# Update application
docker-compose pull
docker-compose up -d --no-deps stock-predictor

# Backup data
docker-compose exec postgres pg_dump -U stockuser stockdb > backup.sql

# Monitor resources
docker stats

# Stop services
docker-compose down

# Clean up
docker system prune -a
```

---

## ☁️ **Cloud Deployment**

### **AWS Deployment**

#### **EC2 Deployment**

**Launch Configuration:**
```bash
#!/bin/bash
# User data script for EC2 instance

# Update system
yum update -y

# Install Docker
amazon-linux-extras install docker
service docker start
usermod -a -G docker ec2-user

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/download/v2.12.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Clone repository
cd /home/ec2-user
git clone https://github.com/your-username/advanced-stock-predictor.git
cd advanced-stock-predictor

# Set up environment
cp .env.production.example .env.production
# Edit .env.production with actual values

# Deploy application
docker-compose up -d

# Set up CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/amazon_linux/amd64/latest/amazon-cloudwatch-agent.rpm
rpm -U ./amazon-cloudwatch-agent.rpm
```

**CloudFormation Template (`infrastructure.yml`):**
```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Advanced Stock Predictor AI Infrastructure'

Parameters:
  KeyPairName:
    Type: AWS::EC2::KeyPair::KeyName
    Description: EC2 Key Pair for SSH access
  
  InstanceType:
    Type: String
    Default: t3.medium
    AllowedValues: [t3.small, t3.medium, t3.large, t3.xlarge]
  
  Environment:
    Type: String
    Default: production
    AllowedValues: [development, staging, production]

Resources:
  # VPC and networking
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: 10.0.0.0/16
      EnableDnsHostnames: true
      EnableDnsSupport: true
      Tags:
        - Key: Name
          Value: !Sub ${Environment}-stock-predictor-vpc

  PublicSubnet:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      CidrBlock: 10.0.1.0/24
      AvailabilityZone: !Select [0, !GetAZs '']
      MapPublicIpOnLaunch: true

  InternetGateway:
    Type: AWS::EC2::InternetGateway

  AttachGateway:
    Type: AWS::EC2::VPCGatewayAttachment
    Properties:
      VpcId: !Ref VPC
      InternetGatewayId: !Ref InternetGateway

  # Security Groups
  ApplicationSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Security group for Stock Predictor application
      VpcId: !Ref VPC
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 80
          ToPort: 80
          CidrIp: 0.0.0.0/0
        - IpProtocol: tcp
          FromPort: 443
          ToPort: 443
          CidrIp: 0.0.0.0/0
        - IpProtocol: tcp
          FromPort: 22
          ToPort: 22
          CidrIp: 0.0.0.0/0

  # Application Load Balancer
  ApplicationLoadBalancer:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Type: application
      Scheme: internet-facing
      SecurityGroups:
        - !Ref ApplicationSecurityGroup
      Subnets:
        - !Ref PublicSubnet

  # Auto Scaling Group
  LaunchTemplate:
    Type: AWS::EC2::LaunchTemplate
    Properties:
      LaunchTemplateName: !Sub ${Environment}-stock-predictor-template
      LaunchTemplateData:
        ImageId: ami-0c02fb55956c7d316  # Amazon Linux 2
        InstanceType: !Ref InstanceType
        KeyName: !Ref KeyPairName
        SecurityGroupIds:
          - !Ref ApplicationSecurityGroup
        UserData:
          Fn::Base64: !Sub |
            #!/bin/bash
            # User data script here

  AutoScalingGroup:
    Type: AWS::AutoScaling::AutoScalingGroup
    Properties:
      LaunchTemplate:
        LaunchTemplateId: !Ref LaunchTemplate
        Version: !GetAtt LaunchTemplate.LatestVersionNumber
      MinSize: 1
      MaxSize: 5
      DesiredCapacity: 2
      VPCZoneIdentifier:
        - !Ref PublicSubnet
      TargetGroupARNs:
        - !Ref TargetGroup

  # RDS Database
  DatabaseSubnetGroup:
    Type: AWS::RDS::DBSubnetGroup
    Properties:
      DBSubnetGroupDescription: Subnet group for RDS database
      SubnetIds:
        - !Ref PublicSubnet

  Database:
    Type: AWS::RDS::DBInstance
    Properties:
      DBInstanceIdentifier: !Sub ${Environment}-stock-predictor-db
      DBInstanceClass: db.t3.micro
      Engine: postgres
      EngineVersion: '14.9'
      MasterUsername: stockuser
      MasterUserPassword: !Ref DatabasePassword
      AllocatedStorage: 20
      StorageType: gp2
      VPCSecurityGroups:
        - !Ref DatabaseSecurityGroup
      DBSubnetGroupName: !Ref DatabaseSubnetGroup

Outputs:
  LoadBalancerDNS:
    Description: DNS name of the load balancer
    Value: !GetAtt ApplicationLoadBalancer.DNSName
  
  DatabaseEndpoint:
    Description: RDS database endpoint
    Value: !GetAtt Database.Endpoint.Address
```

#### **ECS Deployment**

**Task Definition (`ecs-task-definition.json`):**
```json
{
  "family": "stock-predictor-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::ACCOUNT-ID:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT-ID:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "stock-predictor",
      "image": "your-account.dkr.ecr.region.amazonaws.com/stock-predictor:latest",
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        },
        {
          "name": "DATABASE_URL",
          "value": "postgresql://stockuser:password@rds-endpoint:5432/stockdb"
        }
      ],
      "secrets": [
        {
          "name": "SECRET_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:stock-predictor/secret-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/stock-predictor",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": [
          "CMD-SHELL",
          "curl -f http://localhost:8501/_stcore/health || exit 1"
        ],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

### **Google Cloud Platform Deployment**

#### **Cloud Run Deployment**

**`cloudbuild.yaml`:**
```yaml
steps:
  # Build Docker image
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/stock-predictor:$COMMIT_SHA', '.']
  
  # Push to Container Registry
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/stock-predictor:$COMMIT_SHA']
  
  # Deploy to Cloud Run
  - name: 'gcr.io/cloud-builders/gcloud'
    args:
    - 'run'
    - 'deploy'
    - 'stock-predictor'
    - '--image'
    - 'gcr.io/$PROJECT_ID/stock-predictor:$COMMIT_SHA'
    - '--region'
    - 'us-central1'
    - '--platform'
    - 'managed'
    - '--allow-unauthenticated'
    - '--memory'
    - '2Gi'
    - '--cpu'
    - '2'
    - '--max-instances'
    - '10'
    - '--set-env-vars'
    - 'ENVIRONMENT=production'

images:
  - 'gcr.io/$PROJECT_ID/stock-predictor:$COMMIT_SHA'
```

**Deployment Script:**
```bash
#!/bin/bash
# deploy-gcp.sh

# Set project and region
gcloud config set project your-project-id
gcloud config set run/region us-central1

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable sql-component.googleapis.com

# Create Cloud SQL instance
gcloud sql instances create stock-predictor-db \
    --database-version=POSTGRES_14 \
    --tier=db-f1-micro \
    --region=us-central1

# Create database
gcloud sql databases create stockdb --instance=stock-predictor-db

# Deploy application
gcloud builds submit --config cloudbuild.yaml

# Set up domain mapping (optional)
gcloud run domain-mappings create \
    --service stock-predictor \
    --domain your-domain.com
```

### **Azure Deployment**

#### **Container Instances**

**`azuredeploy.json`:**
```json
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "containerGroupName": {
      "type": "string",
      "defaultValue": "stock-predictor-group"
    },
    "containerName": {
      "type": "string",
      "defaultValue": "stock-predictor"
    },
    "image": {
      "type": "string",
      "defaultValue": "your-registry.azurecr.io/stock-predictor:latest"
    }
  },
  "resources": [
    {
      "type": "Microsoft.ContainerInstance/containerGroups",
      "apiVersion": "2021-03-01",
      "name": "[parameters('containerGroupName')]",
      "location": "[resourceGroup().location]",
      "properties": {
        "containers": [
          {
            "name": "[parameters('containerName')]",
            "properties": {
              "image": "[parameters('image')]",
              "ports": [
                {
                  "port": 8501,
                  "protocol": "TCP"
                }
              ],
              "environmentVariables": [
                {
                  "name": "ENVIRONMENT",
                  "value": "production"
                }
              ],
              "resources": {
                "requests": {
                  "cpu": 1,
                  "memoryInGB": 2
                }
              }
            }
          }
        ],
        "osType": "Linux",
        "ipAddress": {
          "type": "Public",
          "ports": [
            {
              "port": 8501,
              "protocol": "TCP"
            }
          ]
        }
      }
    }
  ]
}
```

---

## 🔧 **Configuration Management**

### **Environment-Specific Configurations**

#### **Configuration Factory Pattern**

```python
# config/settings.py
import os
from typing import Dict, Any
from pydantic import BaseSettings, Field

class BaseConfig(BaseSettings):
    """Base configuration"""
    app_name: str = "Advanced Stock Predictor AI"
    version: str = "1.0.0"
    debug: bool = False
    
    # Database
    database_url: str = Field(..., env="DATABASE_URL")
    
    # Redis
    redis_url: str = Field(..., env="REDIS_URL")
    
    # External APIs
    yahoo_finance_timeout: int = Field(30, env="YAHOO_FINANCE_TIMEOUT")
    max_requests_per_minute: int = Field(60, env="MAX_REQUESTS_PER_MINUTE")
    
    # Security
    secret_key: str = Field(..., env="SECRET_KEY")
    session_timeout: int = Field(3600, env="SESSION_TIMEOUT")
    
    class Config:
        env_file = ".env"

class DevelopmentConfig(BaseConfig):
    """Development configuration"""
    debug: bool = True
    log_level: str = "DEBUG"
    cache_ttl: int = 300  # 5 minutes
    
    class Config:
        env_file = ".env.development"

class ProductionConfig(BaseConfig):
    """Production configuration"""
    debug: bool = False
    log_level: str = "INFO"
    cache_ttl: int = 1800  # 30 minutes
    
    # Production-specific settings
    cors_origins: str = Field(..., env="CORS_ORIGINS")
    sentry_dsn: str = Field("", env="SENTRY_DSN")
    
    class Config:
        env_file = ".env.production"

class TestingConfig(BaseConfig):
    """Testing configuration"""
    debug: bool = True
    log_level: str = "DEBUG"
    database_url: str = "sqlite:///test.db"
    redis_url: str = "redis://localhost:6379/1"
    
    class Config:
        env_file = ".env.testing"

def get_config() -> BaseConfig:
    """Get configuration based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    config_map = {
        "development": DevelopmentConfig,
        "production": ProductionConfig,
        "testing": TestingConfig
    }
    
    config_class = config_map.get(env, DevelopmentConfig)
    return config_class()

# Usage
config = get_config()
```

### **Feature Flags**

```python
# config/feature_flags.py
import os
from typing import Dict, Any
import json

class FeatureFlags:
    def __init__(self):
        self.flags = self._load_flags()
    
    def _load_flags(self) -> Dict[str, Any]:
        """Load feature flags from environment or config file"""
        # From environment variable
        flags_json = os.getenv("FEATURE_FLAGS", "{}")
        try:
            return json.loads(flags_json)
        except json.JSONDecodeError:
            return self._get_default_flags()
    
    def _get_default_flags(self) -> Dict[str, Any]:
        """Default feature flags"""
        return {
            "advanced_ml_models": True,
            "real_time_updates": True,
            "social_sentiment": False,
            "crypto_support": False,
            "portfolio_tracking": True,
            "alerts_system": True,
            "mobile_app": False,
            "premium_features": False
        }
    
    def is_enabled(self, flag_name: str) -> bool:
        """Check if feature flag is enabled"""
        return self.flags.get(flag_name, False)
    
    def get_flag(self, flag_name: str, default: Any = None) -> Any:
        """Get feature flag value"""
        return self.flags.get(flag_name, default)

# Usage
feature_flags = FeatureFlags()

if feature_flags.is_enabled("advanced_ml_models"):
    # Load advanced ML models
    pass
```

---

## 📊 **Monitoring & Observability**

### **Application Monitoring**

#### **Prometheus Metrics**

```python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
from functools import wraps

# Define metrics
REQUEST_COUNT = Counter(
    'app_requests_total', 
    'Total app requests', 
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'app_request_duration_seconds', 
    'Request duration in seconds'
)

ACTIVE_USERS = Gauge(
    'app_active_users', 
    'Number of active users'
)

MODEL_PREDICTIONS = Counter(
    'ml_predictions_total', 
    'Total ML predictions made', 
    ['model_type']
)

DATA_FETCH_ERRORS = Counter(
    'data_fetch_errors_total', 
    'Total data fetch errors', 
    ['source']
)

def track_request_metrics(func):
    """Decorator to track request metrics"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            REQUEST_COUNT.labels(
                method='GET', 
                endpoint=func.__name__, 
                status='success'
            ).inc()
            return result
        except Exception as e:
            REQUEST_COUNT.labels(
                method='GET', 
                endpoint=func.__name__, 
                status='error'
            ).inc()
            raise
        finally:
            REQUEST_LATENCY.observe(time.time() - start_time)
    
    return wrapper

# Start metrics server
def start_metrics_server(port=8000):
    start_http_server(port)
```

#### **Grafana Dashboard Configuration**

**`monitoring/grafana/dashboard.json`:**
```json
{
  "dashboard": {
    "title": "Stock Predictor AI Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(app_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, app_request_duration_seconds_bucket)",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Active Users",
        "type": "singlestat",
        "targets": [
          {
            "expr": "app_active_users",
            "legendFormat": "Users"
          }
        ]
      },
      {
        "title": "ML Predictions",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(ml_predictions_total[5m])",
            "legendFormat": "Predictions/sec"
          }
        ]
      }
    ]
  }
}
```

### **Log Management**

#### **Structured Logging**

```python
# monitoring/logging_config.py
import logging
import json
from datetime import datetime
from typing import Dict, Any

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)

def setup_logging(log_level="INFO"):
    """Setup application logging"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app.log')
        ]
    )
    
    # Set JSON formatter
    for handler in logging.root.handlers:
        handler.setFormatter(JSONFormatter())
    
    # Configure specific loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
```

#### **ELK Stack Configuration**

**`monitoring/logstash/logstash.conf`:**
```ruby
input {
  file {
    path => "/app/logs/*.log"
    start_position => "beginning"
    codec => json
  }
}

filter {
  if [level] == "ERROR" {
    mutate {
      add_tag => ["error"]
    }
  }
  
  if [module] == "ml_predictor" {
    mutate {
      add_tag => ["ml"]
    }
  }
  
  date {
    match => [ "timestamp", "ISO8601" ]
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "stock-predictor-%{+YYYY.MM.dd}"
  }
  
  if "error" in [tags] {
    email {
      to => "alerts@yourcompany.com"
      subject => "Stock Predictor Error Alert"
      body => "Error occurred: %{message}"
    }
  }
}
```

---

## 🔒 **Security Configuration**

### **SSL/TLS Setup**

#### **Nginx SSL Configuration**

**`nginx/nginx.conf`:**
```nginx
events {
    worker_connections 1024;
}

http {
    upstream stock_predictor {
        server stock-predictor:8501;
    }
    
    # HTTP to HTTPS redirect
    server {
        listen 80;
        server_name yourdomain.com www.yourdomain.com;
        return 301 https://$server_name$request_uri;
    }
    
    # HTTPS server
    server {
        listen 443 ssl http2;
        server_name yourdomain.com www.yourdomain.com;
        
        # SSL configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;
        ssl_session_cache shared:SSL:10m;
        ssl_session_timeout 10m;
        
        # Security headers
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
        add_header X-Content-Type-Options nosniff;
        add_header X-Frame-Options DENY;
        add_header X-XSS-Protection "1; mode=block";
        add_header Referrer-Policy "strict-origin-when-cross-origin";
        
        # Proxy configuration
        location / {
            proxy_pass http://stock_predictor;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket support
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            
            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }
        
        # Rate limiting
        limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
        limit_req zone=api burst=20 nodelay;
        
        # File upload size
        client_max_body_size 10M;
        
        # Gzip compression
        gzip on;
        gzip_vary on;
        gzip_min_length 1024;
        gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json;
    }
}
```

### **Certificate Management**

**Let's Encrypt with Certbot:**
```bash
#!/bin/bash
# ssl-setup.sh

# Install certbot
apt-get update
apt-get install -y certbot python3-certbot-nginx

# Obtain SSL certificate
certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Set up auto-renewal
echo "0 12 * * * /usr/bin/certbot renew --quiet" | crontab -

# Test renewal
certbot renew --dry-run
```

---

## 🚀 **CI/CD Pipeline**

### **GitHub Actions Workflow**

**`.github/workflows/deploy.yml`:**
```yaml
name: Deploy to Production

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.12'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ --cov=. --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Run security scan
      uses: securecodewarrior/github-action-add-sarif@v1
      with:
        sarif-file: 'security-scan.sarif'

  build:
    needs: [test, security]
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    
    steps:
    - name: Checkout repository
      uses: actions/checkout@v3
    
    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix={{branch}}-
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        build-args: |
          BUILD_DATE=${{ github.event.head_commit.timestamp }}
          VERSION=${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
    - name: Deploy to production
      uses: appleboy/ssh-action@v0.1.5
      with:
        host: ${{ secrets.PROD_HOST }}
        username: ${{ secrets.PROD_USER }}
        key: ${{ secrets.PROD_SSH_KEY }}
        script: |
          cd /home/deploy/stock-predictor
          docker-compose pull
          docker-compose up -d --no-deps stock-predictor
          docker system prune -f
    
    - name: Health check
      run: |
        sleep 30
        curl -f ${{ secrets.PROD_URL }}/_stcore/health || exit 1
    
    - name: Notify deployment
      uses: 8398a7/action-slack@v3
      with:
        status: ${{ job.status }}
        webhook_url: ${{ secrets.SLACK_WEBHOOK }}
      if: always()
```

---

This comprehensive deployment guide covers everything from local development to enterprise-grade production deployments. Choose the deployment strategy that best fits your infrastructure requirements and scale needs.

---

*For additional deployment scenarios or specific cloud provider configurations, refer to the platform-specific documentation.*
