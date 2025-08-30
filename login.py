# login.py
import streamlit as st
import pyrebase

# ========================
# Firebase Configuration
# ========================
firebaseConfig = {
    "apiKey": "AIzaSyCiDlWkYlCgfYJYsQVMNPrLwLmFUaUUaf0",
    "authDomain": "ewde-23aa4.firebaseapp.com",
    "projectId": "ewde-23aa4",
    "storageBucket": "ewde-23aa4.firebasestorage.app",
    "messagingSenderId": "453306204148",
    "appId": "1:453306204148:web:3e5c9069ab3d31fdbcc3a1",
    "measurementId": "",
    "databaseURL": ""
}

try:
    firebase = pyrebase.initialize_app(firebaseConfig)
    auth = firebase.auth()
except Exception as e:
    st.error(f"Firebase initialization error: {str(e)}")
    auth = None

# ========================
# Login Page Styles
# ========================
def apply_login_styles():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        
        /* Main background with animated gradient */
        .stApp {
            background: linear-gradient(-45deg, #0f0f23, #1a1a2e, #16213e, #0f0f23);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
            font-family: 'Poppins', sans-serif;
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Floating particles effect */
        .stApp::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: 
                radial-gradient(circle at 25% 25%, #00f5ff 2px, transparent 2px),
                radial-gradient(circle at 75% 75%, #ff6b6b 1px, transparent 1px),
                radial-gradient(circle at 50% 50%, #4ecdc4 1.5px, transparent 1.5px);
            background-size: 50px 50px, 30px 30px, 40px 40px;
            animation: float 20s ease-in-out infinite;
            opacity: 0.1;
            z-index: -1;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-20px) rotate(180deg); }
        }
        
        /* Main title styling */
        .main-title {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 4rem;
            font-weight: 700;
            text-align: center;
            margin: 3rem 0 2rem 0;
            text-shadow: 0 0 30px rgba(102, 126, 234, 0.5);
            animation: glow 2s ease-in-out infinite alternate;
        }
        
        @keyframes glow {
            from { text-shadow: 0 0 20px rgba(102, 126, 234, 0.5); }
            to { text-shadow: 0 0 30px rgba(102, 126, 234, 0.8); }
        }
        
        /* Subtitle */
        .subtitle {
            text-align: center;
            color: #8892b0;
            font-size: 1.3rem;
            margin-bottom: 3rem;
            font-weight: 300;
        }
        
        /* Login container */
        .login-container {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 25px;
            padding: 3rem 2rem;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
            margin: 2rem auto;
            max-width: 500px;
            transition: all 0.3s ease;
        }
        
        .login-container:hover {
            transform: translateY(-5px);
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.4);
        }
        
        /* Form styling */
        .stTextInput > div > div > input {
            background: rgba(255, 255, 255, 0.1) !important;
            border: 2px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 12px !important;
            color: white !important;
            padding: 1rem !important;
            font-size: 1rem !important;
            transition: all 0.3s ease !important;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #667eea !important;
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.3) !important;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #667eea, #764ba2) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 1rem 2rem !important;
            font-weight: 600 !important;
            font-size: 1.1rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
            width: 100% !important;
            margin-top: 1rem !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.6) !important;
        }
        
        /* Radio button styling */
        .stRadio > div {
            background: rgba(255, 255, 255, 0.05);
            padding: 1rem;
            border-radius: 15px;
            margin-bottom: 2rem;
        }
        
        .stRadio > div > label {
            display: flex !important;
            justify-content: center !important;
            gap: 2rem !important;
        }
        
        /* Success/Error messages */
        .stSuccess, .stError, .stInfo {
            border-radius: 10px !important;
            margin: 1rem 0 !important;
        }
        
        /* Feature highlights */
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 2rem 0;
        }
        
        .feature-card {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 15px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            background: rgba(255, 255, 255, 0.12);
        }
        
        .feature-icon {
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }
        
        .feature-title {
            color: #667eea;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        .feature-desc {
            color: #8892b0;
            font-size: 0.9rem;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# ========================
# Login Page Function
# ========================
def show_login_page():
    """Display the login/signup page"""
    
    # Apply styles
    apply_login_styles()
    
    # Initialize session state
    if "user" not in st.session_state:
        st.session_state.user = None
    if "login_error" not in st.session_state:
        st.session_state.login_error = None

    # Main title and subtitle
    st.markdown('<h1 class="main-title">🚀 Stock Predictor AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced Technical Analysis & AI-Powered Predictions</p>', unsafe_allow_html=True)

    # Feature highlights
    st.markdown('''
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Real-Time Data</div>
            <div class="feature-desc">Live stock prices and market data</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <div class="feature-title">AI Predictions</div>
            <div class="feature-desc">Machine learning price forecasting</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📈</div>
            <div class="feature-title">Technical Analysis</div>
            <div class="feature-desc">RSI, MACD, Moving Averages</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Fast & Secure</div>
            <div class="feature-desc">Lightning-fast analysis</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        
        # Option selection
        option = st.radio(
            "Choose your access method:",
            ["🚀 Demo Mode", "🔑 Login", "📝 Sign Up"],
            horizontal=True,
            help="Demo mode gives full access without registration"
        )

        # Demo Mode
        if option == "🚀 Demo Mode":
            st.markdown("### 🎯 Try Without Registration")
            st.info("🌟 Experience the full power of our AI stock predictor instantly!")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("✅ **Full Access**")
                st.markdown("✅ **AI Predictions**")
                st.markdown("✅ **Technical Analysis**")
            with col_b:
                st.markdown("✅ **Real-time Data**")
                st.markdown("✅ **All Features**")
                st.markdown("✅ **No Registration**")
            
            if st.button("🚀 Launch Demo Mode", key="demo_btn"):
                st.session_state.user = "demo@stockpredictor.ai"
                st.session_state.user_type = "demo"
                st.success("🎉 Welcome to Stock Predictor AI Demo!")
                st.balloons()
                return True

        # Login
        elif option == "🔑 Login":
            with st.form("login_form", clear_on_submit=False):
                st.markdown("### 🔑 Welcome Back!")
                
                email = st.text_input(
                    "📧 Email Address", 
                    placeholder="Enter your email address",
                    help="Use your registered email address"
                )
                
                password = st.text_input(
                    "🔒 Password", 
                    type="password", 
                    placeholder="Enter your password",
                    help="Enter your account password"
                )
                
                col_login, col_forgot = st.columns([2, 1])
                with col_login:
                    submit = st.form_submit_button("🚀 Sign In")
                with col_forgot:
                    forgot = st.form_submit_button("❓ Forgot?")

            if submit and email and password:
                if auth is None:
                    st.error("❌ Authentication service unavailable. Please try Demo Mode.")
                else:
                    try:
                        with st.spinner("🔄 Signing you in..."):
                            user = auth.sign_in_with_email_and_password(email, password)
                            st.session_state.user = user["email"]
                            st.session_state.user_type = "registered"
                            st.success("✅ Login successful! Welcome back!")
                            st.balloons()
                            return True
                    except Exception as e:
                        error_msg = str(e)
                        if "INVALID_PASSWORD" in error_msg:
                            st.error("❌ Incorrect password. Please try again.")
                        elif "EMAIL_NOT_FOUND" in error_msg:
                            st.error("❌ Email not found. Please sign up first.")
                        elif "INVALID_EMAIL" in error_msg:
                            st.error("❌ Please enter a valid email address.")
                        elif "TOO_MANY_ATTEMPTS" in error_msg:
                            st.error("❌ Too many failed attempts. Please try again later.")
                        else:
                            st.error("❌ Login failed. Please check your credentials.")
            
            if forgot:
                st.info("🔄 Password reset functionality coming soon!")

        # Sign Up
        else:  # Sign Up
            with st.form("signup_form", clear_on_submit=False):
                st.markdown("### 📝 Create Your Account")
                
                email = st.text_input(
                    "📧 Email Address", 
                    placeholder="Enter your email address",
                    key="signup_email",
                    help="We'll never share your email"
                )
                
                password = st.text_input(
                    "🔒 Password", 
                    type="password", 
                    placeholder="Create a strong password (min 6 chars)",
                    key="signup_password",
                    help="Use at least 6 characters with mix of letters and numbers"
                )
                
                confirm_password = st.text_input(
                    "🔒 Confirm Password", 
                    type="password", 
                    placeholder="Re-enter your password",
                    help="Must match the password above"
                )
                
                # Terms checkbox
                agree_terms = st.checkbox(
                    "I agree to the Terms of Service and Privacy Policy",
                    help="Required to create an account"
                )
                
                signup = st.form_submit_button("📝 Create Account")

            if signup and email and password:
                if not agree_terms:
                    st.error("❌ Please agree to the Terms of Service to continue.")
                elif len(password) < 6:
                    st.error("❌ Password must be at least 6 characters long.")
                elif password != confirm_password:
                    st.error("❌ Passwords do not match. Please try again.")
                elif auth is None:
                    st.error("❌ Registration service unavailable. Please try Demo Mode.")
                else:
                    try:
                        with st.spinner("🔄 Creating your account..."):
                            auth.create_user_with_email_and_password(email, password)
                            st.success("✅ Account created successfully! Please log in above.")
                            st.balloons()
                    except Exception as e:
                        error_msg = str(e)
                        if "EMAIL_EXISTS" in error_msg:
                            st.error("❌ Email already registered. Please use the login option above.")
                        elif "WEAK_PASSWORD" in error_msg:
                            st.error("❌ Password is too weak. Please use a stronger password.")
                        elif "INVALID_EMAIL" in error_msg:
                            st.error("❌ Please enter a valid email address.")
                        else:
                            st.error("❌ Registration failed. Please try again.")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #8892b0; padding: 2rem;">
        <p style="font-size: 0.9rem;">🔒 Your data is secure and encrypted</p>
        <p style="font-size: 0.8rem;">By using this service, you agree to our Terms of Service and Privacy Policy</p>
    </div>
    """, unsafe_allow_html=True)
    
    return False

# ========================
# Logout Function
# ========================
def logout():
    """Handle user logout"""
    for key in ["user", "user_type", "login_error"]:
        if key in st.session_state:
            del st.session_state[key]
    st.success("👋 Successfully logged out!")
    st.rerun()

# ========================
# Check Authentication Status
# ========================
def is_authenticated():
    """Check if user is logged in"""
    return "user" in st.session_state and st.session_state.user is not None

def get_current_user():
    """Get current user info"""
    if is_authenticated():
        return {
            "email": st.session_state.user,
            "type": st.session_state.get("user_type", "unknown")
        }
    return None