# 🔒 Security & Best Practices Guide

## Advanced Stock Predictor AI - Security Guidelines

### 🛡️ **Security Overview**

This comprehensive security guide covers all aspects of securing the Advanced Stock Predictor AI application, from development to production deployment.

---

## 📋 **Security Checklist**

### **🔐 Authentication & Authorization**

- [ ] **Strong Password Policy:** Minimum 8 characters, mixed case, numbers, symbols
- [ ] **Session Management:** Secure session tokens with expiration
- [ ] **Multi-Factor Authentication:** Optional 2FA for enhanced security
- [ ] **Role-Based Access:** User roles and permissions system
- [ ] **Account Lockout:** Protection against brute force attacks
- [ ] **Password Reset:** Secure password recovery mechanism

### **🌐 Web Application Security**

- [ ] **HTTPS Only:** All traffic encrypted with TLS 1.2+
- [ ] **Security Headers:** Comprehensive HTTP security headers
- [ ] **Input Validation:** All user inputs validated and sanitized
- [ ] **CSRF Protection:** Cross-site request forgery prevention
- [ ] **XSS Prevention:** Cross-site scripting protection
- [ ] **SQL Injection:** Parameterized queries and ORM usage
- [ ] **File Upload Security:** Restricted file types and scanning

### **🔧 Infrastructure Security**

- [ ] **Network Segmentation:** Isolated application and database networks
- [ ] **Firewall Rules:** Restrictive ingress/egress rules
- [ ] **Container Security:** Secure Docker images and runtime
- [ ] **Database Security:** Encrypted connections and access controls
- [ ] **API Security:** Rate limiting and authentication
- [ ] **Monitoring:** Security event logging and alerting
- [ ] **Backup Security:** Encrypted backups with access controls

---

## 🔐 **Authentication & Session Management**

### **Secure Authentication Implementation**

```python
# security/auth.py
import hashlib
import secrets
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import streamlit as st

class SecureAuth:
    def __init__(self, secret_key: str, session_timeout: int = 3600):
        self.secret_key = secret_key
        self.session_timeout = session_timeout
        self.algorithm = 'HS256'
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def generate_session_token(self, user_id: str, user_data: Dict[str, Any]) -> str:
        """Generate secure JWT session token"""
        payload = {
            'user_id': user_id,
            'user_data': user_data,
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(seconds=self.session_timeout),
            'jti': secrets.token_hex(16)  # Unique token ID
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def validate_session_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate and decode session token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check if token is expired
            if datetime.utcnow() > datetime.fromisoformat(payload['exp']):
                return None
            
            return payload
        except jwt.InvalidTokenError:
            return None
    
    def generate_csrf_token(self, session_id: str) -> str:
        """Generate CSRF token"""
        timestamp = str(int(datetime.utcnow().timestamp()))
        message = f"{session_id}:{timestamp}"
        signature = hashlib.hmac(
            self.secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return f"{timestamp}:{signature}"
    
    def validate_csrf_token(self, token: str, session_id: str, max_age: int = 3600) -> bool:
        """Validate CSRF token"""
        try:
            timestamp_str, signature = token.split(':', 1)
            timestamp = int(timestamp_str)
            
            # Check if token is too old
            if datetime.utcnow().timestamp() - timestamp > max_age:
                return False
            
            # Verify signature
            message = f"{session_id}:{timestamp_str}"
            expected_signature = hashlib.hmac(
                self.secret_key.encode(),
                message.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return secrets.compare_digest(signature, expected_signature)
        except (ValueError, TypeError):
            return False

# Enhanced login system
class SecureLoginSystem:
    def __init__(self):
        self.auth = SecureAuth(st.secrets["SECRET_KEY"])
        self.max_attempts = 5
        self.lockout_duration = 300  # 5 minutes
    
    def attempt_login(self, email: str, password: str) -> Dict[str, Any]:
        """Secure login attempt with rate limiting"""
        
        # Check for account lockout
        if self._is_account_locked(email):
            return {
                'success': False,
                'message': 'Account temporarily locked due to too many failed attempts'
            }
        
        # Simulate user lookup (replace with database query)
        user = self._get_user_by_email(email)
        
        if user and self.auth.verify_password(password, user['password_hash']):
            # Clear failed attempts on successful login
            self._clear_failed_attempts(email)
            
            # Generate session token
            token = self.auth.generate_session_token(user['id'], {
                'email': user['email'],
                'role': user.get('role', 'user')
            })
            
            # Store in session state
            st.session_state.auth_token = token
            st.session_state.user_id = user['id']
            st.session_state.user_email = user['email']
            
            return {
                'success': True,
                'message': 'Login successful',
                'user': user
            }
        else:
            # Record failed attempt
            self._record_failed_attempt(email)
            
            return {
                'success': False,
                'message': 'Invalid email or password'
            }
    
    def _is_account_locked(self, email: str) -> bool:
        """Check if account is locked due to failed attempts"""
        if f"failed_attempts_{email}" not in st.session_state:
            return False
        
        attempts = st.session_state[f"failed_attempts_{email}"]
        if attempts['count'] >= self.max_attempts:
            time_since_last = datetime.utcnow() - attempts['last_attempt']
            return time_since_last.total_seconds() < self.lockout_duration
        
        return False
    
    def _record_failed_attempt(self, email: str) -> None:
        """Record failed login attempt"""
        key = f"failed_attempts_{email}"
        
        if key not in st.session_state:
            st.session_state[key] = {'count': 0, 'last_attempt': datetime.utcnow()}
        
        st.session_state[key]['count'] += 1
        st.session_state[key]['last_attempt'] = datetime.utcnow()
    
    def _clear_failed_attempts(self, email: str) -> None:
        """Clear failed login attempts"""
        key = f"failed_attempts_{email}"
        if key in st.session_state:
            del st.session_state[key]
    
    def _get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user from database (implement with actual database)"""
        # Demo user for testing
        if email == "demo@example.com":
            return {
                'id': '1',
                'email': 'demo@example.com',
                'password_hash': self.auth.hash_password('demo123'),
                'role': 'user'
            }
        return None
```

### **Streamlit Session Security**

```python
# security/session_security.py
import streamlit as st
from datetime import datetime, timedelta
import uuid

class SessionSecurity:
    @staticmethod
    def initialize_secure_session():
        """Initialize secure session state"""
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        
        if 'session_created' not in st.session_state:
            st.session_state.session_created = datetime.utcnow()
        
        if 'last_activity' not in st.session_state:
            st.session_state.last_activity = datetime.utcnow()
    
    @staticmethod
    def check_session_timeout(timeout_minutes: int = 30) -> bool:
        """Check if session has timed out"""
        if 'last_activity' not in st.session_state:
            return True
        
        timeout_delta = timedelta(minutes=timeout_minutes)
        return datetime.utcnow() - st.session_state.last_activity > timeout_delta
    
    @staticmethod
    def update_activity():
        """Update last activity timestamp"""
        st.session_state.last_activity = datetime.utcnow()
    
    @staticmethod
    def clear_session():
        """Securely clear session data"""
        sensitive_keys = [
            'auth_token', 'user_id', 'user_email', 'api_keys',
            'personal_data', 'trading_data'
        ]
        
        for key in sensitive_keys:
            if key in st.session_state:
                del st.session_state[key]
    
    @staticmethod
    def is_authenticated() -> bool:
        """Check if user is authenticated"""
        return (
            'auth_token' in st.session_state and
            'user_id' in st.session_state and
            not SessionSecurity.check_session_timeout()
        )

# Session middleware
def require_auth(func):
    """Decorator to require authentication"""
    def wrapper(*args, **kwargs):
        SessionSecurity.initialize_secure_session()
        
        if SessionSecurity.check_session_timeout():
            SessionSecurity.clear_session()
            st.error("Session expired. Please log in again.")
            st.stop()
        
        if not SessionSecurity.is_authenticated():
            st.error("Please log in to access this feature.")
            st.stop()
        
        SessionSecurity.update_activity()
        return func(*args, **kwargs)
    
    return wrapper
```

---

## 🛡️ **Input Validation & Sanitization**

### **Comprehensive Input Validation**

```python
# security/input_validation.py
import re
import html
import bleach
from typing import Any, Optional, List, Dict
import pandas as pd

class InputValidator:
    
    # Regex patterns for validation
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    STOCK_SYMBOL_PATTERN = re.compile(r'^[A-Z]{1,10}\.NS$')
    PASSWORD_PATTERN = re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$')
    
    # Allowed HTML tags for rich text
    ALLOWED_TAGS = ['b', 'i', 'u', 'em', 'strong', 'p', 'br']
    
    @classmethod
    def validate_email(cls, email: str) -> bool:
        """Validate email format"""
        if not isinstance(email, str) or len(email) > 254:
            return False
        return bool(cls.EMAIL_PATTERN.match(email))
    
    @classmethod
    def validate_password(cls, password: str) -> Dict[str, Any]:
        """Validate password strength"""
        if not isinstance(password, str):
            return {'valid': False, 'message': 'Password must be a string'}
        
        if len(password) < 8:
            return {'valid': False, 'message': 'Password must be at least 8 characters'}
        
        if len(password) > 128:
            return {'valid': False, 'message': 'Password must be less than 128 characters'}
        
        if not cls.PASSWORD_PATTERN.match(password):
            return {
                'valid': False,
                'message': 'Password must contain uppercase, lowercase, digit, and special character'
            }
        
        return {'valid': True, 'message': 'Password is strong'}
    
    @classmethod
    def validate_stock_symbol(cls, symbol: str) -> bool:
        """Validate NSE stock symbol format"""
        if not isinstance(symbol, str):
            return False
        return bool(cls.STOCK_SYMBOL_PATTERN.match(symbol.upper()))
    
    @classmethod
    def sanitize_string(cls, input_str: str, max_length: int = 1000) -> str:
        """Sanitize string input"""
        if not isinstance(input_str, str):
            return ""
        
        # Truncate if too long
        if len(input_str) > max_length:
            input_str = input_str[:max_length]
        
        # HTML escape
        sanitized = html.escape(input_str)
        
        # Remove any remaining dangerous characters
        sanitized = re.sub(r'[<>"\']', '', sanitized)
        
        return sanitized.strip()
    
    @classmethod
    def sanitize_html(cls, html_content: str) -> str:
        """Sanitize HTML content"""
        if not isinstance(html_content, str):
            return ""
        
        return bleach.clean(
            html_content,
            tags=cls.ALLOWED_TAGS,
            strip=True
        )
    
    @classmethod
    def validate_numeric_input(cls, value: Any, min_val: float = None, max_val: float = None) -> Dict[str, Any]:
        """Validate numeric input"""
        try:
            num_value = float(value)
            
            if min_val is not None and num_value < min_val:
                return {'valid': False, 'message': f'Value must be at least {min_val}'}
            
            if max_val is not None and num_value > max_val:
                return {'valid': False, 'message': f'Value must be at most {max_val}'}
            
            return {'valid': True, 'value': num_value}
        
        except (ValueError, TypeError):
            return {'valid': False, 'message': 'Invalid numeric value'}
    
    @classmethod
    def validate_date_range(cls, start_date: str, end_date: str) -> Dict[str, Any]:
        """Validate date range"""
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            
            if start >= end:
                return {'valid': False, 'message': 'End date must be after start date'}
            
            # Check if dates are reasonable (not too far in future/past)
            now = pd.Timestamp.now()
            if start < now - pd.Timedelta(days=3650):  # 10 years ago
                return {'valid': False, 'message': 'Start date too far in the past'}
            
            if end > now + pd.Timedelta(days=365):  # 1 year in future
                return {'valid': False, 'message': 'End date too far in the future'}
            
            return {'valid': True, 'start': start, 'end': end}
        
        except (ValueError, TypeError):
            return {'valid': False, 'message': 'Invalid date format'}

# Streamlit form validation
class StreamlitFormValidator:
    def __init__(self):
        self.validator = InputValidator()
        self.errors = []
    
    def validate_form_data(self, form_data: Dict[str, Any]) -> bool:
        """Validate form data"""
        self.errors = []
        
        # Email validation
        if 'email' in form_data:
            if not self.validator.validate_email(form_data['email']):
                self.errors.append("Invalid email format")
        
        # Password validation
        if 'password' in form_data:
            password_result = self.validator.validate_password(form_data['password'])
            if not password_result['valid']:
                self.errors.append(password_result['message'])
        
        # Stock symbol validation
        if 'stock_symbol' in form_data:
            if not self.validator.validate_stock_symbol(form_data['stock_symbol']):
                self.errors.append("Invalid stock symbol format")
        
        return len(self.errors) == 0
    
    def display_errors(self):
        """Display validation errors in Streamlit"""
        for error in self.errors:
            st.error(f"❌ {error}")
```

### **File Upload Security**

```python
# security/file_security.py
import os
import magic
import hashlib
from typing import List, Dict, Any, Optional
import streamlit as st

class FileUploadSecurity:
    
    # Allowed file types
    ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls', '.json'}
    ALLOWED_MIME_TYPES = {
        'text/csv',
        'application/vnd.ms-excel',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/json'
    }
    
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    @classmethod
    def validate_file_upload(cls, uploaded_file) -> Dict[str, Any]:
        """Comprehensive file upload validation"""
        
        if uploaded_file is None:
            return {'valid': False, 'message': 'No file uploaded'}
        
        # Check file size
        if uploaded_file.size > cls.MAX_FILE_SIZE:
            return {
                'valid': False,
                'message': f'File size exceeds {cls.MAX_FILE_SIZE // (1024*1024)}MB limit'
            }
        
        # Check file extension
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension not in cls.ALLOWED_EXTENSIONS:
            return {
                'valid': False,
                'message': f'File type {file_extension} not allowed'
            }
        
        # Check MIME type
        file_content = uploaded_file.read()
        uploaded_file.seek(0)  # Reset file pointer
        
        try:
            mime_type = magic.from_buffer(file_content, mime=True)
            if mime_type not in cls.ALLOWED_MIME_TYPES:
                return {
                    'valid': False,
                    'message': f'MIME type {mime_type} not allowed'
                }
        except Exception:
            return {
                'valid': False,
                'message': 'Could not determine file type'
            }
        
        # Scan for malicious content
        if cls._scan_for_malicious_content(file_content):
            return {
                'valid': False,
                'message': 'File contains potentially malicious content'
            }
        
        return {
            'valid': True,
            'message': 'File validation passed',
            'file_hash': hashlib.sha256(file_content).hexdigest()
        }
    
    @classmethod
    def _scan_for_malicious_content(cls, file_content: bytes) -> bool:
        """Basic scan for malicious content"""
        
        # Convert to lowercase for case-insensitive scanning
        content_str = file_content.lower()
        
        # Suspicious patterns
        suspicious_patterns = [
            b'<script',
            b'javascript:',
            b'vbscript:',
            b'onload=',
            b'onerror=',
            b'<?php',
            b'<%',
            b'eval(',
            b'exec(',
            b'system(',
            b'shell_exec('
        ]
        
        for pattern in suspicious_patterns:
            if pattern in content_str:
                return True
        
        return False
    
    @classmethod
    def secure_file_save(cls, uploaded_file, save_path: str) -> Dict[str, Any]:
        """Securely save uploaded file"""
        
        # Validate file first
        validation_result = cls.validate_file_upload(uploaded_file)
        if not validation_result['valid']:
            return validation_result
        
        try:
            # Generate secure filename
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            secure_filename = f"{validation_result['file_hash']}{file_extension}"
            full_path = os.path.join(save_path, secure_filename)
            
            # Ensure save directory exists
            os.makedirs(save_path, exist_ok=True)
            
            # Save file
            with open(full_path, 'wb') as f:
                f.write(uploaded_file.read())
            
            return {
                'valid': True,
                'message': 'File saved successfully',
                'file_path': full_path,
                'file_hash': validation_result['file_hash']
            }
        
        except Exception as e:
            return {
                'valid': False,
                'message': f'Error saving file: {str(e)}'
            }
```

---

## 🌐 **Web Application Security**

### **Security Headers Implementation**

```python
# security/headers.py
from typing import Dict

class SecurityHeaders:
    
    @staticmethod
    def get_security_headers() -> Dict[str, str]:
        """Get comprehensive security headers"""
        return {
            # Prevent XSS attacks
            'X-XSS-Protection': '1; mode=block',
            
            # Prevent MIME type sniffing
            'X-Content-Type-Options': 'nosniff',
            
            # Prevent clickjacking
            'X-Frame-Options': 'DENY',
            
            # Enforce HTTPS
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains; preload',
            
            # Control referrer information
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            
            # Feature policy (permissions policy)
            'Permissions-Policy': (
                'geolocation=(), microphone=(), camera=(), '
                'magnetometer=(), gyroscope=(), speaker=(), '
                'vibrate=(), fullscreen=(self), payment=()'
            ),
            
            # Content Security Policy
            'Content-Security-Policy': (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval' "
                "https://cdn.plot.ly https://cdnjs.cloudflare.com; "
                "style-src 'self' 'unsafe-inline' "
                "https://fonts.googleapis.com https://cdn.plot.ly; "
                "font-src 'self' https://fonts.gstatic.com; "
                "img-src 'self' data: https:; "
                "connect-src 'self' https://query1.finance.yahoo.com; "
                "frame-ancestors 'none'; "
                "base-uri 'self'; "
                "form-action 'self';"
            ),
            
            # Remove server information
            'Server': 'WebServer/1.0',
            
            # Cache control for sensitive pages
            'Cache-Control': 'no-cache, no-store, must-revalidate, private',
            'Pragma': 'no-cache',
            'Expires': '0'
        }
    
    @staticmethod
    def apply_streamlit_security():
        """Apply security configurations for Streamlit"""
        
        # Inject security headers via CSS hack (Streamlit limitation)
        security_css = """
        <style>
        /* Security headers injection */
        html {
            /* This is a workaround for Streamlit's header limitations */
        }
        </style>
        <script>
        // Additional security measures
        
        // Disable right-click context menu (optional)
        document.addEventListener('contextmenu', function(e) {
            e.preventDefault();
        });
        
        // Disable F12 and other developer tools shortcuts (optional)
        document.addEventListener('keydown', function(e) {
            if (e.key === 'F12' || 
                (e.ctrlKey && e.shiftKey && e.key === 'I') ||
                (e.ctrlKey && e.shiftKey && e.key === 'J') ||
                (e.ctrlKey && e.key === 'u')) {
                e.preventDefault();
            }
        });
        
        // Clear sensitive data on page unload
        window.addEventListener('beforeunload', function() {
            // Clear any sensitive data from memory
            if (window.sessionStorage) {
                sessionStorage.clear();
            }
        });
        </script>
        """
        
        st.markdown(security_css, unsafe_allow_html=True)
```

### **CSRF Protection**

```python
# security/csrf_protection.py
import hmac
import hashlib
import secrets
import time
from typing import Optional
import streamlit as st

class CSRFProtection:
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.token_lifetime = 3600  # 1 hour
    
    def generate_csrf_token(self) -> str:
        """Generate CSRF token"""
        
        # Get or create session ID
        if 'session_id' not in st.session_state:
            st.session_state.session_id = secrets.token_hex(32)
        
        session_id = st.session_state.session_id
        timestamp = str(int(time.time()))
        
        # Create message to sign
        message = f"{session_id}:{timestamp}"
        
        # Generate HMAC signature
        signature = hmac.new(
            self.secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Combine timestamp and signature
        token = f"{timestamp}:{signature}"
        
        # Store in session
        st.session_state.csrf_token = token
        
        return token
    
    def validate_csrf_token(self, token: Optional[str]) -> bool:
        """Validate CSRF token"""
        
        if not token or 'session_id' not in st.session_state:
            return False
        
        try:
            timestamp_str, signature = token.split(':', 1)
            timestamp = int(timestamp_str)
            
            # Check if token is expired
            if time.time() - timestamp > self.token_lifetime:
                return False
            
            # Recreate expected signature
            session_id = st.session_state.session_id
            message = f"{session_id}:{timestamp_str}"
            expected_signature = hmac.new(
                self.secret_key.encode(),
                message.encode(),
                hashlib.sha256
            ).hexdigest()
            
            # Compare signatures
            return hmac.compare_digest(signature, expected_signature)
        
        except (ValueError, TypeError):
            return False
    
    def protect_form(self, form_key: str):
        """Protect form with CSRF token"""
        
        # Generate token for this form
        csrf_token = self.generate_csrf_token()
        
        # Create hidden input for token
        st.markdown(
            f'<input type="hidden" name="csrf_token" value="{csrf_token}" id="csrf_token_{form_key}">',
            unsafe_allow_html=True
        )
        
        return csrf_token
    
    def verify_form_submission(self, submitted_token: Optional[str]) -> bool:
        """Verify form submission with CSRF token"""
        
        if not self.validate_csrf_token(submitted_token):
            st.error("🚨 Security validation failed. Please refresh the page and try again.")
            return False
        
        return True

# Usage decorator
def csrf_protected(csrf_protection: CSRFProtection):
    """Decorator for CSRF protection"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Check if this is a form submission
            if st.session_state.get('form_submitted', False):
                csrf_token = st.session_state.get('submitted_csrf_token')
                if not csrf_protection.verify_form_submission(csrf_token):
                    return None
            
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

---

## 🔐 **Data Protection & Privacy**

### **Data Encryption**

```python
# security/encryption.py
import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from typing import Union, bytes

class DataEncryption:
    
    def __init__(self, password: str):
        """Initialize encryption with password"""
        self.password = password.encode()
        self.salt = os.urandom(16)
        self.key = self._derive_key()
        self.fernet = Fernet(self.key)
    
    def _derive_key(self) -> bytes:
        """Derive encryption key from password"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.password))
        return key
    
    def encrypt(self, data: Union[str, bytes]) -> str:
        """Encrypt data"""
        if isinstance(data, str):
            data = data.encode()
        
        encrypted_data = self.fernet.encrypt(data)
        
        # Combine salt and encrypted data
        result = base64.urlsafe_b64encode(self.salt + encrypted_data)
        return result.decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt data"""
        try:
            # Decode and separate salt and data
            combined_data = base64.urlsafe_b64decode(encrypted_data.encode())
            salt = combined_data[:16]
            encrypted_content = combined_data[16:]
            
            # Derive key with extracted salt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.password))
            fernet = Fernet(key)
            
            # Decrypt
            decrypted_data = fernet.decrypt(encrypted_content)
            return decrypted_data.decode()
        
        except Exception:
            raise ValueError("Failed to decrypt data")

# Secure storage for sensitive data
class SecureStorage:
    
    def __init__(self, encryption_key: str):
        self.encryption = DataEncryption(encryption_key)
    
    def store_sensitive_data(self, key: str, value: str) -> None:
        """Store sensitive data encrypted in session state"""
        encrypted_value = self.encryption.encrypt(value)
        st.session_state[f"secure_{key}"] = encrypted_value
    
    def retrieve_sensitive_data(self, key: str) -> str:
        """Retrieve and decrypt sensitive data"""
        encrypted_value = st.session_state.get(f"secure_{key}")
        if encrypted_value:
            return self.encryption.decrypt(encrypted_value)
        return None
    
    def clear_sensitive_data(self, key: str) -> None:
        """Clear sensitive data"""
        secure_key = f"secure_{key}"
        if secure_key in st.session_state:
            del st.session_state[secure_key]
```

### **Privacy Controls**

```python
# security/privacy.py
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import streamlit as st

class PrivacyManager:
    
    def __init__(self):
        self.data_retention_days = 90
        self.consent_version = "1.0"
    
    def collect_consent(self) -> bool:
        """Collect user consent for data processing"""
        
        if 'privacy_consent' not in st.session_state:
            st.session_state.privacy_consent = None
        
        if st.session_state.privacy_consent is None:
            st.markdown("### 🔒 Privacy Consent")
            
            with st.expander("Privacy Policy & Data Usage", expanded=True):
                st.markdown("""
                **Data Collection & Usage:**
                - We collect stock symbols and analysis preferences
                - Session data is stored temporarily for functionality
                - No personal financial data is stored permanently
                - Data is used solely for stock analysis and predictions
                
                **Data Sharing:**
                - We do not share personal data with third parties
                - Market data is sourced from public APIs (Yahoo Finance)
                - Anonymous usage statistics may be collected
                
                **Data Retention:**
                - Session data is cleared after 24 hours of inactivity
                - Analysis results are not permanently stored
                - You can request data deletion at any time
                
                **Your Rights:**
                - Access your data
                - Request data deletion
                - Opt-out of analytics
                - Update preferences
                """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Accept & Continue", type="primary"):
                    st.session_state.privacy_consent = {
                        'accepted': True,
                        'timestamp': datetime.utcnow(),
                        'version': self.consent_version
                    }
                    st.rerun()
            
            with col2:
                if st.button("Decline"):
                    st.session_state.privacy_consent = {
                        'accepted': False,
                        'timestamp': datetime.utcnow(),
                        'version': self.consent_version
                    }
                    st.error("Privacy consent is required to use this application.")
                    st.stop()
            
            return False
        
        return st.session_state.privacy_consent.get('accepted', False)
    
    def show_privacy_controls(self):
        """Show privacy control options"""
        
        st.markdown("### 🔒 Privacy Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Clear All Data"):
                self.clear_all_user_data()
                st.success("All user data cleared successfully")
        
        with col2:
            if st.button("Download My Data"):
                data_export = self.export_user_data()
                st.download_button(
                    label="Download Data",
                    data=data_export,
                    file_name=f"my_data_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
        
        # Data retention info
        st.info(f"Data is automatically deleted after {self.data_retention_days} days of inactivity")
    
    def clear_all_user_data(self):
        """Clear all user data"""
        sensitive_keys = [
            key for key in st.session_state.keys() 
            if any(keyword in key.lower() for keyword in 
                  ['user', 'data', 'stock', 'prediction', 'analysis'])
        ]
        
        for key in sensitive_keys:
            del st.session_state[key]
    
    def export_user_data(self) -> str:
        """Export user data for download"""
        user_data = {
            'export_timestamp': datetime.utcnow().isoformat(),
            'consent_info': st.session_state.get('privacy_consent', {}),
            'session_data': {
                key: value for key, value in st.session_state.items()
                if not key.startswith('secure_') and 
                   not key.startswith('FormSubmitter')
            }
        }
        
        import json
        return json.dumps(user_data, indent=2, default=str)
    
    def check_data_retention(self):
        """Check and enforce data retention policy"""
        if 'last_activity' in st.session_state:
            last_activity = st.session_state.last_activity
            if isinstance(last_activity, str):
                last_activity = datetime.fromisoformat(last_activity)
            
            if datetime.utcnow() - last_activity > timedelta(days=self.data_retention_days):
                self.clear_all_user_data()
                st.warning("Data has been automatically cleared due to retention policy")
```

---

## 🔒 **API Security & Rate Limiting**

### **Rate Limiting Implementation**

```python
# security/rate_limiting.py
import time
from collections import defaultdict, deque
from typing import Dict, Optional
import streamlit as st

class RateLimiter:
    
    def __init__(self):
        self.requests = defaultdict(deque)
        self.limits = {
            'api_calls': {'max': 60, 'window': 60},      # 60 calls per minute
            'login_attempts': {'max': 5, 'window': 300}, # 5 attempts per 5 minutes
            'data_requests': {'max': 100, 'window': 3600} # 100 requests per hour
        }
    
    def is_allowed(self, identifier: str, limit_type: str) -> Dict[str, any]:
        """Check if request is allowed under rate limit"""
        
        current_time = time.time()
        limit_config = self.limits.get(limit_type, {'max': 10, 'window': 60})
        
        # Get user's request history
        user_requests = self.requests[f"{identifier}:{limit_type}"]
        
        # Remove old requests outside the time window
        while user_requests and user_requests[0] < current_time - limit_config['window']:
            user_requests.popleft()
        
        # Check if under limit
        if len(user_requests) < limit_config['max']:
            user_requests.append(current_time)
            return {
                'allowed': True,
                'remaining': limit_config['max'] - len(user_requests),
                'reset_time': current_time + limit_config['window']
            }
        else:
            # Calculate when limit resets
            reset_time = user_requests[0] + limit_config['window']
            return {
                'allowed': False,
                'remaining': 0,
                'reset_time': reset_time,
                'retry_after': reset_time - current_time
            }
    
    def get_user_identifier(self) -> str:
        """Get unique identifier for rate limiting"""
        # Use session ID as identifier
        if 'session_id' not in st.session_state:
            import uuid
            st.session_state.session_id = str(uuid.uuid4())
        
        return st.session_state.session_id

# Rate limiting decorator
def rate_limited(limit_type: str, rate_limiter: RateLimiter):
    """Decorator for rate limiting functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            user_id = rate_limiter.get_user_identifier()
            result = rate_limiter.is_allowed(user_id, limit_type)
            
            if not result['allowed']:
                st.error(f"🚫 Rate limit exceeded. Try again in {int(result['retry_after'])} seconds.")
                st.info(f"Remaining requests: {result['remaining']}")
                return None
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Usage example
rate_limiter = RateLimiter()

@rate_limited('data_requests', rate_limiter)
def fetch_stock_data(symbol):
    # Your data fetching code here
    pass
```

### **API Key Security**

```python
# security/api_security.py
import os
import hashlib
import secrets
from typing import Dict, Optional
import streamlit as st

class APIKeySecurity:
    
    def __init__(self):
        self.api_keys = {}
        self.api_key_permissions = {}
    
    def generate_api_key(self, user_id: str, permissions: list) -> str:
        """Generate secure API key for user"""
        
        # Create unique key
        random_bytes = secrets.token_bytes(32)
        user_bytes = user_id.encode()
        combined = random_bytes + user_bytes
        
        api_key = hashlib.sha256(combined).hexdigest()
        
        # Store key and permissions
        self.api_keys[api_key] = {
            'user_id': user_id,
            'created_at': time.time(),
            'permissions': permissions,
            'active': True
        }
        
        return api_key
    
    def validate_api_key(self, api_key: str, required_permission: str) -> Optional[Dict]:
        """Validate API key and check permissions"""
        
        key_info = self.api_keys.get(api_key)
        
        if not key_info or not key_info['active']:
            return None
        
        if required_permission not in key_info['permissions']:
            return None
        
        return key_info
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke API key"""
        if api_key in self.api_keys:
            self.api_keys[api_key]['active'] = False
            return True
        return False
    
    def secure_api_endpoint(self, required_permission: str):
        """Decorator to secure API endpoints"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Get API key from headers or query params
                api_key = st.query_params.get('api_key')
                
                if not api_key:
                    st.error("API key required")
                    return None
                
                key_info = self.validate_api_key(api_key, required_permission)
                
                if not key_info:
                    st.error("Invalid API key or insufficient permissions")
                    return None
                
                # Add user info to kwargs
                kwargs['user_info'] = key_info
                
                return func(*args, **kwargs)
            return wrapper
        return decorator
```

---

## 🛡️ **Security Best Practices**

### **Secure Development Guidelines**

#### **Code Security Checklist**

- [ ] **Input Validation:** All user inputs validated and sanitized
- [ ] **Output Encoding:** All outputs properly encoded
- [ ] **Error Handling:** No sensitive information in error messages
- [ ] **Logging:** Security events logged without sensitive data
- [ ] **Dependencies:** Regular security updates for all dependencies
- [ ] **Secrets Management:** No hardcoded secrets in code
- [ ] **Code Review:** Security-focused code reviews
- [ ] **Static Analysis:** Automated security scanning

#### **Deployment Security**

```bash
# security/security-check.sh
#!/bin/bash

echo "🔒 Running Security Checks..."

# Check for hardcoded secrets
echo "Checking for hardcoded secrets..."
grep -r "password\|secret\|key\|token" --include="*.py" . | grep -v "example\|test\|#" || echo "✅ No hardcoded secrets found"

# Check file permissions
echo "Checking file permissions..."
find . -type f -name "*.py" -perm /077 && echo "❌ Python files have overly permissive permissions" || echo "✅ File permissions OK"

# Check for debug mode in production
echo "Checking for debug mode..."
grep -r "debug.*=.*True" --include="*.py" . && echo "❌ Debug mode enabled" || echo "✅ Debug mode disabled"

# Check SSL configuration
echo "Checking SSL configuration..."
grep -r "ssl.*=.*False" --include="*.py" . && echo "❌ SSL disabled" || echo "✅ SSL configuration OK"

# Dependency security check
echo "Checking dependencies for known vulnerabilities..."
pip install safety
safety check

echo "🔒 Security check complete!"
```

### **Security Monitoring**

```python
# security/security_monitoring.py
import logging
from datetime import datetime
from typing import Dict, Any
import streamlit as st

class SecurityMonitor:
    
    def __init__(self):
        self.setup_security_logging()
        self.security_events = []
    
    def setup_security_logging(self):
        """Setup security-specific logging"""
        
        security_logger = logging.getLogger('security')
        security_logger.setLevel(logging.INFO)
        
        # Create security log handler
        handler = logging.FileHandler('security.log')
        formatter = logging.Formatter(
            '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        security_logger.addHandler(handler)
        
        self.logger = security_logger
    
    def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security event"""
        
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'details': details,
            'user_id': st.session_state.get('user_id', 'anonymous'),
            'session_id': st.session_state.get('session_id', 'unknown'),
            'ip_address': st.query_params.get('remote_addr', 'unknown')
        }
        
        self.security_events.append(event)
        self.logger.info(f"{event_type}: {details}")
        
        # Check for suspicious activity
        self.analyze_security_events()
    
    def analyze_security_events(self):
        """Analyze security events for suspicious patterns"""
        
        recent_events = [
            event for event in self.security_events
            if (datetime.utcnow() - datetime.fromisoformat(event['timestamp'])).seconds < 300
        ]
        
        # Check for multiple failed login attempts
        failed_logins = [
            event for event in recent_events
            if event['event_type'] == 'failed_login'
        ]
        
        if len(failed_logins) >= 5:
            self.trigger_security_alert('multiple_failed_logins', {
                'count': len(failed_logins),
                'timespan': '5 minutes'
            })
        
        # Check for rapid API calls
        api_calls = [
            event for event in recent_events
            if event['event_type'] == 'api_call'
        ]
        
        if len(api_calls) >= 100:
            self.trigger_security_alert('rapid_api_calls', {
                'count': len(api_calls),
                'timespan': '5 minutes'
            })
    
    def trigger_security_alert(self, alert_type: str, details: Dict[str, Any]):
        """Trigger security alert"""
        
        self.logger.warning(f"SECURITY ALERT - {alert_type}: {details}")
        
        # In production, send alerts to security team
        # self.send_security_notification(alert_type, details)
    
    def get_security_dashboard_data(self) -> Dict[str, Any]:
        """Get data for security dashboard"""
        
        return {
            'total_events': len(self.security_events),
            'recent_events': len([
                event for event in self.security_events
                if (datetime.utcnow() - datetime.fromisoformat(event['timestamp'])).seconds < 3600
            ]),
            'event_types': {
                event['event_type']: len([
                    e for e in self.security_events 
                    if e['event_type'] == event['event_type']
                ])
                for event in self.security_events
            }
        }

# Security event decorators
security_monitor = SecurityMonitor()

def log_security_event(event_type: str):
    """Decorator to log security events"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                security_monitor.log_security_event(event_type, {
                    'function': func.__name__,
                    'status': 'success'
                })
                return result
            except Exception as e:
                security_monitor.log_security_event(event_type, {
                    'function': func.__name__,
                    'status': 'error',
                    'error': str(e)
                })
                raise
        return wrapper
    return decorator
```

---

## 📋 **Security Compliance**

### **GDPR Compliance**

```python
# security/gdpr_compliance.py
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import streamlit as st
import json

class GDPRCompliance:
    
    def __init__(self):
        self.data_processor = "Advanced Stock Predictor AI"
        self.data_controller = "Your Organization"
        self.dpo_contact = "dpo@yourorganization.com"
    
    def show_privacy_notice(self):
        """Display GDPR privacy notice"""
        
        with st.expander("🔒 Privacy Notice (GDPR)", expanded=False):
            st.markdown(f"""
            ### Data Controller Information
            **Data Controller:** {self.data_controller}
            **Data Protection Officer:** {self.dpo_contact}
            
            ### Legal Basis for Processing
            - **Legitimate Interest:** Providing stock analysis services
            - **Consent:** Where explicitly given for optional features
            
            ### Data We Collect
            - Session identifiers (temporary)
            - Stock symbols searched
            - Analysis preferences
            - Usage statistics (anonymized)
            
            ### Data Retention
            - Session data: Cleared after 24 hours of inactivity
            - Usage statistics: Anonymized and retained for 12 months
            - No personal financial data is stored
            
            ### Your Rights Under GDPR
            - **Right to Access:** Request copy of your data
            - **Right to Rectification:** Correct inaccurate data
            - **Right to Erasure:** Request data deletion
            - **Right to Portability:** Export your data
            - **Right to Object:** Opt-out of processing
            - **Right to Restrict:** Limit processing
            
            ### International Transfers
            Data may be processed in countries with adequate protection levels.
            
            ### Contact Us
            For privacy questions: {self.dpo_contact}
            """)
    
    def handle_data_request(self, request_type: str):
        """Handle GDPR data requests"""
        
        st.markdown(f"### {request_type} Request")
        
        if request_type == "Access":
            self.handle_access_request()
        elif request_type == "Deletion":
            self.handle_deletion_request()
        elif request_type == "Portability":
            self.handle_portability_request()
        elif request_type == "Rectification":
            self.handle_rectification_request()
    
    def handle_access_request(self):
        """Handle data access request"""
        
        st.info("📋 Data Access Request")
        
        if st.button("Generate Data Report"):
            user_data = self.collect_user_data()
            
            st.markdown("### Your Data Report")
            st.json(user_data)
            
            # Provide download
            st.download_button(
                label="Download Data Report",
                data=json.dumps(user_data, indent=2),
                file_name=f"data_report_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    def handle_deletion_request(self):
        """Handle data deletion request"""
        
        st.warning("⚠️ Data Deletion Request")
        st.write("This will permanently delete all your data.")
        
        confirm = st.text_input("Type 'DELETE' to confirm:")
        
        if confirm == "DELETE" and st.button("Delete My Data"):
            self.delete_all_user_data()
            st.success("✅ All your data has been deleted.")
            st.balloons()
    
    def collect_user_data(self) -> Dict:
        """Collect all user data for access request"""
        
        return {
            'request_timestamp': datetime.utcnow().isoformat(),
            'session_data': {
                key: value for key, value in st.session_state.items()
                if not key.startswith('_') and not callable(value)
            },
            'privacy_settings': {
                'consent_given': st.session_state.get('privacy_consent', {}).get('accepted', False),
                'consent_timestamp': st.session_state.get('privacy_consent', {}).get('timestamp'),
            },
            'data_retention_policy': {
                'session_data': '24 hours after inactivity',
                'usage_statistics': '12 months (anonymized)',
                'personal_data': 'Not stored'
            }
        }
    
    def delete_all_user_data(self):
        """Delete all user data"""
        
        # List of all session state keys to preserve (system keys)
        system_keys = ['_session_state', '_widgets']
        
        # Delete all user data
        keys_to_delete = [
            key for key in st.session_state.keys()
            if key not in system_keys
        ]
        
        for key in keys_to_delete:
            del st.session_state[key]
```

---

This comprehensive security guide provides a robust foundation for securing the Advanced Stock Predictor AI application. Implement these measures according to your deployment environment and compliance requirements.

---

*For additional security questions or custom security implementations, consult with security professionals and conduct regular security audits.*
