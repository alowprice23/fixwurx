#!/usr/bin/env python3
"""
Web Interface Module

This module provides a web-based interface to the FixWurx shell environment,
allowing users to access the shell through a web browser.
"""

import os
import sys
import json
import time
import logging
import threading
import secrets
import uuid
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path
from datetime import datetime, timedelta

# Web server imports
from flask import (
    Flask, render_template, request, jsonify, redirect,
    url_for, session, flash, send_from_directory
)
from flask_socketio import SocketIO, emit, join_room, leave_room
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import eventlet

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("web_interface.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("WebInterface")

# Define constants
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000
SESSION_LIFETIME = 3600  # 1 hour
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'txt', 'py', 'json', 'yaml', 'yml', 'md', 'log', 'csv', 'xml', 'html', 'css', 'js', 'sh', 'bat', 'fx'}

class WebInterface:
    """Web interface for the FixWurx shell environment."""
    
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        """
        Initialize the web interface.
        
        Args:
            host: Host to bind to
            port: Port to listen on
        """
        self.host = host
        self.port = port
        self.registry = None
        self.shell_env = None
        self.users = {}
        self.sessions = {}
        self.shell_instances = {}
        self.running = False
        self.server = None
        self.socketio = None
        self.app = self._create_app()
    
    def _create_app(self):
        """
        Create the Flask application.
        
        Returns:
            Flask application
        """
        # Create Flask app
        app = Flask(__name__, 
                   template_folder="templates/web_interface", 
                   static_folder="static/web_interface")
        app.config['SECRET_KEY'] = secrets.token_hex(32)
        app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(seconds=SESSION_LIFETIME)
        app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
        
        # Create SocketIO instance
        self.socketio = SocketIO(app, async_mode='eventlet')
        
        # Configure routes
        self._configure_routes(app)
        
        # Configure socket events
        self._configure_socket_events()
        
        return app
    
    def _configure_routes(self, app):
        """Configure Flask routes."""
        
        @app.route('/')
        def index():
            """Render the index page."""
            if 'user_id' not in session:
                return redirect(url_for('login'))
            
            return render_template('index.html', 
                                  username=session.get('username', 'User'),
                                  session_id=session.get('session_id'))
        
        @app.route('/login', methods=['GET', 'POST'])
        def login():
            """Handle login requests."""
            if request.method == 'POST':
                username = request.form.get('username')
                password = request.form.get('password')
                
                if self._authenticate_user(username, password):
                    # Generate user ID
                    user_id = str(uuid.uuid4())
                    
                    # Create session
                    session_id = self._create_session(user_id, username)
                    
                    # Store in Flask session
                    session['user_id'] = user_id
                    session['username'] = username
                    session['session_id'] = session_id
                    
                    # Create shell instance
                    self._create_shell_instance(session_id)
                    
                    # Redirect to index
                    return redirect(url_for('index'))
                else:
                    flash('Invalid username or password')
            
            return render_template('login.html')
        
        @app.route('/logout')
        def logout():
            """Handle logout requests."""
            if 'session_id' in session:
                # Remove shell instance
                session_id = session['session_id']
                self._remove_shell_instance(session_id)
                
                # Remove session
                if session_id in self.sessions:
                    del self.sessions[session_id]
            
            # Clear Flask session
            session.clear()
            
            # Redirect to login
            return redirect(url_for('login'))
        
        @app.route('/api/command', methods=['POST'])
        def api_command():
            """Handle command execution requests."""
            if 'user_id' not in session:
                return jsonify({'error': 'Not authenticated'}), 401
            
            # Get session
            session_id = session.get('session_id')
            if session_id not in self.sessions:
                return jsonify({'error': 'Invalid session'}), 403
            
            # Get command
            data = request.json
            command = data.get('command', '').strip()
            
            if not command:
                return jsonify({'error': 'Command required'}), 400
            
            # Execute command
            output, exit_code = self._execute_command(session_id, command)
            
            # Return result
            return jsonify({
                'command': command,
                'output': output,
                'exit_code': exit_code
            })
        
        @app.route('/api/history', methods=['GET'])
        def api_history():
            """Get command history."""
            if 'user_id' not in session:
                return jsonify({'error': 'Not authenticated'}), 401
            
            # Get session
            session_id = session.get('session_id')
            if session_id not in self.sessions:
                return jsonify({'error': 'Invalid session'}), 403
            
            # Get history
            history = self._get_command_history(session_id)
            
            # Return history
            return jsonify({
                'history': history
            })
        
        @app.route('/api/status', methods=['GET'])
        def api_status():
            """Get system status."""
            if 'user_id' not in session:
                return jsonify({'error': 'Not authenticated'}), 401
            
            # Get components
            components = {}
            if self.registry:
                for name, component in self.registry.get_components().items():
                    if hasattr(component, 'get_status') and callable(component.get_status):
                        try:
                            components[name] = component.get_status()
                        except Exception as e:
                            components[name] = {'error': str(e)}
            
            # Return status
            return jsonify({
                'components': components,
                'web_interface': {
                    'uptime': time.time() - self.start_time if hasattr(self, 'start_time') else 0,
                    'sessions': len(self.sessions),
                    'shell_instances': len(self.shell_instances)
                }
            })
        
        @app.route('/upload', methods=['POST'])
        def upload_file():
            """Handle file uploads."""
            if 'user_id' not in session:
                return jsonify({'error': 'Not authenticated'}), 401
            
            # Check if file is present
            if 'file' not in request.files:
                return jsonify({'error': 'No file part'}), 400
            
            file = request.files['file']
            
            # Check if filename is empty
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400
            
            # Check if file extension is allowed
            if file and self._allowed_file(file.filename):
                filename = secure_filename(file.filename)
                
                # Create upload folder if it doesn't exist
                os.makedirs(UPLOAD_FOLDER, exist_ok=True)
                
                # Save file
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(file_path)
                
                return jsonify({
                    'message': 'File uploaded successfully',
                    'filename': filename,
                    'path': file_path
                })
            else:
                return jsonify({'error': 'File type not allowed'}), 400
        
        @app.route('/download/<path:filename>')
        def download_file(filename):
            """Handle file downloads."""
            if 'user_id' not in session:
                return jsonify({'error': 'Not authenticated'}), 401
            
            # Send file
            try:
                return send_from_directory(directory=os.getcwd(), 
                                         filename=filename, 
                                         as_attachment=True)
            except Exception as e:
                return jsonify({'error': f'Error downloading file: {str(e)}'}), 404
    
    def _configure_socket_events(self):
        """Configure SocketIO event handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            if 'user_id' not in session:
                return False
            
            # Get session
            session_id = session.get('session_id')
            if session_id not in self.sessions:
                return False
            
            # Join room
            join_room(session_id)
            
            # Update session
            self.sessions[session_id]['last_activity'] = time.time()
            self.sessions[session_id]['socket_connected'] = True
            
            # Send welcome message
            emit('system_message', {
                'message': f"Welcome {session.get('username', 'User')}! You are now connected to the FixWurx shell environment."
            })
            
            logger.info(f"Socket connected: {session_id}")
            return True
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            if 'user_id' not in session:
                return
            
            # Get session
            session_id = session.get('session_id')
            if session_id not in self.sessions:
                return
            
            # Leave room
            leave_room(session_id)
            
            # Update session
            self.sessions[session_id]['socket_connected'] = False
            
            logger.info(f"Socket disconnected: {session_id}")
        
        @self.socketio.on('execute_command')
        def handle_execute_command(data):
            """Handle command execution requests."""
            if 'user_id' not in session:
                emit('error', {'message': 'Not authenticated'})
                return
            
            # Get session
            session_id = session.get('session_id')
            if session_id not in self.sessions:
                emit('error', {'message': 'Invalid session'})
                return
            
            # Get command
            command = data.get('command', '').strip()
            
            if not command:
                emit('error', {'message': 'Command required'})
                return
            
            # Execute command
            output, exit_code = self._execute_command(session_id, command)
            
            # Send result
            emit('command_result', {
                'command': command,
                'output': output,
                'exit_code': exit_code,
                'timestamp': time.time()
            })
        
        @self.socketio.on('ping')
        def handle_ping():
            """Handle ping requests."""
            emit('pong', {'server_time': time.time()})
        
        @self.socketio.on('join_collaboration')
        def handle_join_collaboration(data):
            """Handle collaboration join requests."""
            if 'user_id' not in session:
                emit('error', {'message': 'Not authenticated'})
                return
            
            # Get collaboration ID
            collaboration_id = data.get('collaboration_id')
            
            if not collaboration_id:
                emit('error', {'message': 'Collaboration ID required'})
                return
            
            # Join collaboration room
            join_room(f"collab_{collaboration_id}")
            
            # Update session
            session_id = session.get('session_id')
            if session_id in self.sessions:
                self.sessions[session_id]['collaboration_id'] = collaboration_id
            
            # Notify other users
            emit('user_joined', {
                'username': session.get('username', 'User'),
                'timestamp': time.time()
            }, room=f"collab_{collaboration_id}", include_self=False)
            
            logger.info(f"User {session.get('username', 'User')} joined collaboration: {collaboration_id}")
        
        @self.socketio.on('leave_collaboration')
        def handle_leave_collaboration():
            """Handle collaboration leave requests."""
            if 'user_id' not in session:
                return
            
            # Get session
            session_id = session.get('session_id')
            if session_id not in self.sessions:
                return
            
            # Get collaboration ID
            collaboration_id = self.sessions[session_id].get('collaboration_id')
            
            if not collaboration_id:
                return
            
            # Leave collaboration room
            leave_room(f"collab_{collaboration_id}")
            
            # Update session
            self.sessions[session_id]['collaboration_id'] = None
            
            # Notify other users
            emit('user_left', {
                'username': session.get('username', 'User'),
                'timestamp': time.time()
            }, room=f"collab_{collaboration_id}")
            
            logger.info(f"User {session.get('username', 'User')} left collaboration: {collaboration_id}")
        
        @self.socketio.on('collaboration_message')
        def handle_collaboration_message(data):
            """Handle collaboration messages."""
            if 'user_id' not in session:
                emit('error', {'message': 'Not authenticated'})
                return
            
            # Get session
            session_id = session.get('session_id')
            if session_id not in self.sessions:
                emit('error', {'message': 'Invalid session'})
                return
            
            # Get collaboration ID
            collaboration_id = self.sessions[session_id].get('collaboration_id')
            
            if not collaboration_id:
                emit('error', {'message': 'Not in a collaboration'})
                return
            
            # Get message
            message = data.get('message', '').strip()
            
            if not message:
                emit('error', {'message': 'Message required'})
                return
            
            # Broadcast message
            emit('collaboration_message', {
                'username': session.get('username', 'User'),
                'message': message,
                'timestamp': time.time()
            }, room=f"collab_{collaboration_id}")
            
            logger.info(f"Collaboration message from {session.get('username', 'User')}: {message}")
    
    def set_registry(self, registry):
        """Set the component registry."""
        self.registry = registry
    
    def set_shell_env(self, shell_env):
        """Set the shell environment."""
        self.shell_env = shell_env
    
    def load_users(self, users_file: str):
        """
        Load users from a file.
        
        Args:
            users_file: Path to users file
        """
        try:
            # Check if file exists
            if not os.path.exists(users_file):
                logger.warning(f"Users file not found: {users_file}")
                return
            
            # Load users
            with open(users_file, 'r') as f:
                users = json.load(f)
            
            # Update users
            self.users.update(users)
            
            logger.info(f"Loaded {len(users)} users from {users_file}")
        except Exception as e:
            logger.error(f"Error loading users: {e}")
    
    def start(self):
        """Start the web interface."""
        if self.running:
            logger.warning("Web interface already running")
            return False
        
        try:
            # Create templates directory if it doesn't exist
            os.makedirs("templates/web_interface", exist_ok=True)
            
            # Create static directory if it doesn't exist
            os.makedirs("static/web_interface", exist_ok=True)
            
            # Create upload directory if it doesn't exist
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            
            # Generate default templates if they don't exist
            self._generate_default_templates()
            
            # Generate default static files if they don't exist
            self._generate_default_static_files()
            
            # Store start time
            self.start_time = time.time()
            
            # Start cleaning thread
            self.cleaning_thread = threading.Thread(target=self._cleaning_thread)
            self.cleaning_thread.daemon = True
            self.cleaning_thread.start()
            
            # Start server
            self.running = True
            logger.info(f"Starting web interface on {self.host}:{self.port}")
            self.socketio.run(self.app, host=self.host, port=self.port)
            
            return True
        except Exception as e:
            logger.error(f"Error starting web interface: {e}")
            return False
    
    def stop(self):
        """Stop the web interface."""
        if not self.running:
            logger.warning("Web interface not running")
            return False
        
        try:
            # Stop server
            self.running = False
            
            # Clean up
            for session_id in list(self.shell_instances.keys()):
                self._remove_shell_instance(session_id)
            
            self.sessions.clear()
            
            logger.info("Web interface stopped")
            return True
        except Exception as e:
            logger.error(f"Error stopping web interface: {e}")
            return False
    
    def _authenticate_user(self, username: str, password: str) -> bool:
        """
        Authenticate a user.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            True if authenticated, False otherwise
        """
        # Check if user exists
        if username not in self.users:
            return False
        
        # Get user
        user = self.users[username]
        
        # Check password
        if 'password_hash' in user:
            return check_password_hash(user['password_hash'], password)
        elif 'password' in user:
            return user['password'] == password
        else:
            return False
    
    def _create_session(self, user_id: str, username: str) -> str:
        """
        Create a session for a user.
        
        Args:
            user_id: User ID
            username: Username
            
        Returns:
            Session ID
        """
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Create session
        self.sessions[session_id] = {
            'user_id': user_id,
            'username': username,
            'created_at': time.time(),
            'last_activity': time.time(),
            'socket_connected': False,
            'collaboration_id': None
        }
        
        return session_id
    
    def _create_shell_instance(self, session_id: str):
        """
        Create a shell instance for a session.
        
        Args:
            session_id: Session ID
        """
        if session_id not in self.sessions:
            logger.error(f"Session not found: {session_id}")
            return
        
        # Create shell instance
        self.shell_instances[session_id] = {
            'cwd': os.getcwd(),
            'environment': {
                'PATH': os.environ.get('PATH', ''),
                'HOME': os.environ.get('HOME', ''),
                'USER': self.sessions[session_id]['username']
            },
            'history': []
        }
    
    def _remove_shell_instance(self, session_id: str):
        """
        Remove a shell instance.
        
        Args:
            session_id: Session ID
        """
        if session_id in self.shell_instances:
            del self.shell_instances[session_id]
    
    def _execute_command(self, session_id: str, command: str) -> tuple:
        """
        Execute a command.
        
        Args:
            session_id: Session ID
            command: Command to execute
            
        Returns:
            Tuple of (output, exit_code)
        """
        if session_id not in self.shell_instances:
            return "Error: Shell instance not found", 1
        
        # Get shell instance
        shell = self.shell_instances[session_id]
        
        # Add to history
        shell['history'].append({
            'command': command,
            'timestamp': time.time()
        })
        
        # Execute command
        try:
            # Split command into parts
            parts = command.split(maxsplit=1)
            if not parts:
                return "", 0
            
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            # Special handling for CD command
            if cmd == 'cd':
                return self._change_directory(session_id, args)
            
            # Find command handler
            if not self.registry:
                return "Error: Registry not available", 1
            
            handler_info = self.registry.get_command_handler(cmd)
            
            if not handler_info:
                return f"Unknown command: {cmd}", 1
            
            # Execute command
            # Redirect stdout to capture output
            original_stdout = sys.stdout
            output_capture = OutputCapture()
            sys.stdout = output_capture
            
            try:
                # Execute command handler
                handler = handler_info["handler"]
                exit_code = handler(args)
                
                # Get captured output
                output = output_capture.get_output()
                
                return output, exit_code
            finally:
                # Restore stdout
                sys.stdout = original_stdout
        except Exception as e:
            return f"Error executing command: {e}", 1
    
    def _change_directory(self, session_id: str, path: str) -> tuple:
        """
        Change directory for a shell instance.
        
        Args:
            session_id: Session ID
            path: Directory path
            
        Returns:
            Tuple of (output, exit_code)
        """
        if session_id not in self.shell_instances:
            return "Error: Shell instance not found", 1
        
        # Get shell instance
        shell = self.shell_instances[session_id]
        
        try:
            # Handle special cases
            if not path:
                path = os.environ.get('HOME', '/')
            elif path == '-':
                # Go to previous directory
                path = shell.get('prev_cwd', shell['cwd'])
            
            # Expand path
            if path.startswith('~'):
                path = os.path.expanduser(path)
            elif not os.path.isabs(path):
                path = os.path.join(shell['cwd'], path)
            
            # Normalize path
            path = os.path.normpath(path)
            
            # Check if directory exists
            if not os.path.isdir(path):
                return f"cd: {path}: No such directory", 1
            
            # Save previous directory
            shell['prev_cwd'] = shell['cwd']
            
            # Change directory
            shell['cwd'] = path
            os.chdir(path)
            
            return "", 0
        except Exception as e:
            return f"cd: {e}", 1
    
    def _get_command_history(self, session_id: str) -> List[Dict]:
        """
        Get command history for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Command history
        """
        if session_id not in self.shell_instances:
            return []
        
        # Get shell instance
        shell = self.shell_instances[session_id]
        
        # Return history
        return shell['history']
    
    def _cleaning_thread(self):
        """Cleaning thread for expired sessions."""
        while self.running:
            try:
                # Get current time
                current_time = time.time()
                
                # Check sessions
                for session_id in list(self.sessions.keys()):
                    session = self.sessions[session_id]
                    
                    # Check if expired
                    if current_time - session['last_activity'] > SESSION_LIFETIME:
                        # Remove shell instance
                        self._remove_shell_instance(session_id)
                        
                        # Remove session
                        del self.sessions[session_id]
                        
                        logger.info(f"Session expired: {session_id}")
            except Exception as e:
                logger.error(f"Error in cleaning thread: {e}")
            
            # Sleep
            time.sleep(60)
    
    def _allowed_file(self, filename: str) -> bool:
        """
        Check if a file extension is allowed.
        
        Args:
            filename: Filename
            
        Returns:
            True if allowed, False otherwise
        """
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    def _generate_default_templates(self):
        """Generate default HTML templates."""
        # Create index.html
        index_path = "templates/web_interface/index.html"
        if not os.path.exists(index_path):
            with open(index_path, 'w') as f:
                f.write('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FixWurx Shell</title>
    <link rel="stylesheet" href="/static/web_interface/css/style.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
</head>
<body>
    <div class="container">
        <header>
            <h1>FixWurx Shell Environment</h1>
            <div class="user-info">
                <span>Welcome, {{ username }}</span>
                <a href="/logout" class="btn btn-sm">Logout</a>
            </div>
        </header>
        
        <div class="main-container">
            <div class="sidebar">
                <div class="sidebar-section">
                    <h3>System Status</h3>
                    <div id="system-status">
                        <p>Loading...</p>
                    </div>
                </div>
                
                <div class="sidebar-section">
                    <h3>Command History</h3>
                    <div id="command-history">
                        <ul class="history-list"></ul>
                    </div>
                </div>
                
                <div class="sidebar-section">
                    <h3>File Operations</h3>
                    <div class="file-ops">
                        <form id="upload-form" enctype="multipart/form-data">
                            <input type="file" id="file-input" name="file">
                            <button type="submit" class="btn btn-sm">Upload</button>
                        </form>
                    </div>
                </div>
                
                <div class="sidebar-section">
                    <h3>Collaboration</h3>
                    <div class="collab-tools">
                        <input type="text" id="collab-id" placeholder="Collaboration ID">
                        <button id="join-collab" class="btn btn-sm">Join</button>
                        <button id="leave-collab" class="btn btn-sm" disabled>Leave</button>
                    </div>
                    <div id="collab-users">
                        <ul class="users-list"></ul>
                    </div>
                    <div id="collab-chat">
                        <div class="chat-messages"></div>
                        <div class="chat-input">
                            <input type="text" id="chat-message" placeholder="Type a message...">
                            <button id="send-message" class="btn btn-sm">Send</button>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="terminal-container">
                <div id="terminal-output"></div>
                <div class="command-input">
                    <span class="prompt">$</span>
                    <input type="text" id="command-input" placeholder="Enter command...">
                </div>
            </div>
        </div>
    </div>
    
    <script src="/static/web_interface/js/script.js"></script>
</body>
</html>''')
        
        # Create login.html
        login_path = "templates/web_interface/login.html"
        if not os.path.exists(login_path):
            with open(login_path, 'w') as f:
                f.write('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login - FixWurx Shell</title>
    <link rel="stylesheet" href="/static/web_interface/css/style.css">
</head>
<body>
    <div class="login-container">
        <div class="login-box">
            <h1>FixWurx Shell</h1>
            <h2>Login</h2>
            
            {% with messages = get_flashed_messages() %}
                {% if messages %}
                    <div class="alert alert-error">
                        {% for message in messages %}
                            {{ message }}
                        {% endfor %}
                    </div>
                {% endif %}
            {% endwith %}
            
            <form method="post" action="/login">
                <div class="form-group">
                    <label for="username">Username</label>
                    <input type="text" id="username" name="username" required>
                </div>
                
                <div class="form-group">
                    <label for="password">Password</label>
                    <input type="password" id="password" name="password" required>
                </div>
                
                <div class="form-group">
                    <button type="submit" class="btn btn-primary">Login</button>
                </div>
            </form>
        </div>
    </div>
</body>
</html>''')
    
    def _generate_default_static_files(self):
        """Generate default static files."""
        # Create CSS directory
        css_dir = "static/web_interface/css"
        os.makedirs(css_dir, exist_ok=True)
        
        # Create style.css
        style_path = f"{css_dir}/style.css"
        if not os.path.exists(style_path):
            with open(style_path, 'w') as f:
                f.write('''/* Reset and base styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: #333;
    background-color: #f8f9fa;
}

.container {
    display: flex;
    flex-direction: column;
    height: 100vh;
}

/* Header styles */
header {
    background-color: #2c3e50;
    color: white;
    padding: 10px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

header h1 {
    font-size: 1.5rem;
}

.user-info {
    display: flex;
    align-items: center;
    gap: 15px;
}

/* Main container styles */
.main-container {
    display: flex;
    flex: 1;
    overflow: hidden;
}

/* Sidebar styles */
.sidebar {
    width: 300px;
    background-color: #ecf0f1;
    border-right: 1px solid #ddd;
    overflow-y: auto;
    padding: 15px;
}

.sidebar-section {
    margin-bottom: 20px;
}

.sidebar-section h3 {
    border-bottom: 1px solid #ddd;
    padding-bottom: 5px;
    margin-bottom: 10px;
    color: #2c3e50;
}

/* Terminal styles */
.terminal-container {
    flex: 1;
    display: flex;
    flex-direction: column;
    background-color: #1e1e1e;
    color: #f0f0f0;
    font-family: 'Courier New', Courier, monospace;
    overflow: hidden;
}

#terminal-output {
    flex: 1;
    padding: 15px;
    overflow-y: auto;
    white-space: pre-wrap;
}

.command-input {
    display: flex;
    padding: 10px 15px;
    background-color: #252525;
    border-top: 1px solid #333;
}

.prompt {
    color: #27ae60;
    margin-right: 10px;
}

#command-input {
    flex: 1;
    background-color: transparent;
    border: none;
    color: #f0f0f0;
    font-family: 'Courier New', Courier, monospace;
    outline: none;
}

/* Command history styles */
.history-list {
    list-style-type: none;
    max-height: 200px;
    overflow-y: auto;
}

.history-list li {
    padding: 5px 0;
    cursor: pointer;
    border-bottom: 1px solid #ddd;
}

.history-list li:hover {
    background-color: #e7e7e7;
}

/* File operations styles */
.file-ops {
    display: flex;
    flex-direction: column;
    gap: 10px;
}

/* Collaboration styles */
.collab-tools {
    display: flex;
    gap: 5px;
    margin-bottom: 10px;
}

#collab-id {
    flex: 1;
    padding: 5px;
    border: 1px solid #ddd;
    border-radius: 3px;
}

.users-list {
    list-style-type: none;
    margin-bottom: 10px;
    max-height: 100px;
    overflow-y: auto;
}

.users-list li {
    padding: 2px 5px;
    background-color: #e7e7e7;
    margin-bottom: 2px;
    border-radius: 3px;
}

#collab-chat {
    display: flex;
    flex-direction: column;
    height: 200px;
    border: 1px solid #ddd;
    border-radius: 3px;
}

.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 5px;
    background-color: white;
}

.chat-input {
    display: flex;
    padding: 5px;
    border-top: 1px solid #ddd;
}

#chat-message {
    flex: 1;
    padding: 5px;
    border: 1px solid #ddd;
    border-radius: 3px;
}

/* Button styles */
.btn {
    padding: 8px 15px;
    background-color: #3498db;
    color: white;
    border: none;
    border-radius: 3px;
    cursor: pointer;
}

.btn:hover {
    background-color: #2980b9;
}

.btn-sm {
    padding: 5px 10px;
    font-size: 0.8rem;
}

.btn-primary {
    background-color: #2980b9;
}

/* Login page styles */
.login-container {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
    background-color: #2c3e50;
}

.login-box {
    width: 400px;
    padding: 30px;
    background-color: white;
    border-radius: 5px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
}

.login-box h1 {
    text-align: center;
    margin-bottom: 5px;
    color: #2c3e50;
}

.login-box h2 {
    text-align: center;
    margin-bottom: 20px;
    color: #7f8c8d;
}

.form-group {
    margin-bottom: 15px;
}

.form-group label {
    display: block;
    margin-bottom: 5px;
}

.form-group input {
    width: 100%;
    padding: 8px;
    border: 1px solid #ddd;
    border-radius: 3px;
}

.alert {
    padding: 10px;
    margin-bottom: 15px;
    border-radius: 3px;
    color: white;
}

.alert-error {
    background-color: #e74c3c;
}

/* Terminal command styles */
.command {
    color: #27ae60;
    margin-top: 5px;
}

.error {
    color: #e74c3c;
}

.success {
    color: #2ecc71;
}

.info {
    color: #3498db;
}

.warning {
    color: #f39c12;
}
''')
        
        # Create JS directory
        js_dir = "static/web_interface/js"
        os.makedirs(js_dir, exist_ok=True)
        
        # Create script.js
        script_path = f"{js_dir}/script.js"
        if not os.path.exists(script_path):
            with open(script_path, 'w') as f:
                f.write('''// Connect to the socket server
const socket = io();

// DOM elements
const terminalOutput = document.getElementById('terminal-output');
const commandInput = document.getElementById('command-input');
const commandHistory = document.getElementById('command-history');
const historyList = document.querySelector('.history-list');
const systemStatus = document.getElementById('system-status');
const uploadForm = document.getElementById('upload-form');
const fileInput = document.getElementById('file-input');
const collabId = document.getElementById('collab-id');
const joinCollab = document.getElementById('join-collab');
const leaveCollab = document.getElementById('leave-collab');
const collabUsers = document.getElementById('collab-users');
const usersList = document.querySelector('.users-list');
const chatMessages = document.querySelector('.chat-messages');
const chatMessage = document.getElementById('chat-message');
const sendMessage = document.getElementById('send-message');

// Variables
let history = [];
let historyIndex = -1;
let inCollaboration = false;
let collaborationId = null;
let users = [];

// Add a command to the terminal
function addCommand(command) {
    const commandElement = document.createElement('div');
    commandElement.classList.add('command');
    commandElement.textContent = `$ ${command}`;
    terminalOutput.appendChild(commandElement);
    terminalOutput.scrollTop = terminalOutput.scrollHeight;
}

// Add output to the terminal
function addOutput(output, exitCode = 0) {
    const outputElement = document.createElement('div');
    outputElement.innerHTML = output;
    
    if (exitCode !== 0) {
        outputElement.classList.add('error');
    }
    
    terminalOutput.appendChild(outputElement);
    terminalOutput.scrollTop = terminalOutput.scrollHeight;
}

// Execute a command
function executeCommand(command) {
    if (!command) return;
    
    addCommand(command);
    
    // Add to history if not already there
    if (history.length === 0 || history[history.length - 1] !== command) {
        history.push(command);
        addToHistoryList(command);
    }
    
    historyIndex = history.length;
    
    // Send to server
    socket.emit('execute_command', { command });
}

// Add a command to the history list
function addToHistoryList(command) {
    const item = document.createElement('li');
    item.textContent = command;
    item.addEventListener('click', () => {
        commandInput.value = command;
        commandInput.focus();
    });
    historyList.appendChild(item);
}

// Load command history
function loadHistory() {
    fetch('/api/history')
        .then(response => response.json())
        .then(data => {
            history = data.history.map(item => item.command);
            historyList.innerHTML = '';
            history.forEach(command => addToHistoryList(command));
            historyIndex = history.length;
        })
        .catch(error => console.error('Error loading history:', error));
}

// Load system status
function loadStatus() {
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            let statusHtml = '<div class="status-grid">';
            
            // Web interface status
            statusHtml += '<div class="status-section">';
            statusHtml += '<h4>Web Interface</h4>';
            statusHtml += `<p>Uptime: ${formatTime(data.web_interface.uptime)}</p>`;
            statusHtml += `<p>Active Sessions: ${data.web_interface.sessions}</p>`;
            statusHtml += '</div>';
            
            // Components status
            for (const [name, status] of Object.entries(data.components)) {
                statusHtml += '<div class="status-section">';
                statusHtml += `<h4>${name}</h4>`;
                
                if (status.error) {
                    statusHtml += `<p class="error">${status.error}</p>`;
                } else {
                    for (const [key, value] of Object.entries(status)) {
                        if (typeof value === 'object') continue;
                        statusHtml += `<p>${key}: ${value}</p>`;
                    }
                }
                
                statusHtml += '</div>';
            }
            
            statusHtml += '</div>';
            systemStatus.innerHTML = statusHtml;
        })
        .catch(error => console.error('Error loading status:', error));
}

// Format time (seconds) to human-readable format
function formatTime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    seconds = Math.floor(seconds % 60);
    
    return `${hours}h ${minutes}m ${seconds}s`;
}

// Format timestamp to human-readable format
function formatTimestamp(timestamp) {
    const date = new Date(timestamp * 1000);
    return date.toLocaleTimeString();
}

// Join collaboration
function joinCollaboration() {
    const id = collabId.value.trim();
    if (!id) return;
    
    socket.emit('join_collaboration', { collaboration_id: id });
    
    inCollaboration = true;
    collaborationId = id;
    joinCollab.disabled = true;
    leaveCollab.disabled = false;
    
    // Add system message
    const systemMsg = document.createElement('div');
    systemMsg.classList.add('system-message');
    systemMsg.textContent = `Joined collaboration: ${id}`;
    chatMessages.appendChild(systemMsg);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Leave collaboration
function leaveCollaboration() {
    if (!inCollaboration) return;
    
    socket.emit('leave_collaboration');
    
    inCollaboration = false;
    collaborationId = null;
    joinCollab.disabled = false;
    leaveCollab.disabled = true;
    usersList.innerHTML = '';
    
    // Add system message
    const systemMsg = document.createElement('div');
    systemMsg.classList.add('system-message');
    systemMsg.textContent = 'Left collaboration';
    chatMessages.appendChild(systemMsg);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Send chat message
function sendChatMessage() {
    if (!inCollaboration) return;
    
    const message = chatMessage.value.trim();
    if (!message) return;
    
    socket.emit('collaboration_message', { message });
    chatMessage.value = '';
}

// Add chat message
function addChatMessage(username, message, timestamp) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('chat-message');
    
    const timeStr = formatTimestamp(timestamp);
    messageElement.innerHTML = `<span class="chat-time">[${timeStr}]</span> <span class="chat-user">${username}:</span> ${message}`;
    
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Add user to collaboration
function addUser(username) {
    if (users.includes(username)) return;
    
    users.push(username);
    
    const userElement = document.createElement('li');
    userElement.textContent = username;
    usersList.appendChild(userElement);
}

// Remove user from collaboration
function removeUser(username) {
    users = users.filter(user => user !== username);
    
    const userElements = usersList.querySelectorAll('li');
    for (const element of userElements) {
        if (element.textContent === username) {
            element.remove();
            break;
        }
    }
}

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    // Focus on command input
    commandInput.focus();
    
    // Load history and status
    loadHistory();
    loadStatus();
    
    // Set up status refresh interval
    setInterval(loadStatus, 30000); // 30 seconds
});

// Command input event listener
commandInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
        const command = commandInput.value.trim();
        if (command) {
            executeCommand(command);
            commandInput.value = '';
        }
    } else if (event.key === 'ArrowUp') {
        // Navigate history up
        if (historyIndex > 0) {
            historyIndex--;
            commandInput.value = history[historyIndex];
            
            // Move cursor to end
            setTimeout(() => {
                commandInput.selectionStart = commandInput.selectionEnd = commandInput.value.length;
            }, 0);
        }
        event.preventDefault();
    } else if (event.key === 'ArrowDown') {
        // Navigate history down
        if (historyIndex < history.length - 1) {
            historyIndex++;
            commandInput.value = history[historyIndex];
        } else if (historyIndex === history.length - 1) {
            historyIndex++;
            commandInput.value = '';
        }
        event.preventDefault();
    }
});

// File upload event listener
uploadForm.addEventListener('submit', (event) => {
    event.preventDefault();
    
    const file = fileInput.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            addOutput(`Error: ${data.error}`, 1);
        } else {
            addOutput(`File uploaded: ${data.filename} to ${data.path}`, 0);
        }
    })
    .catch(error => {
        addOutput(`Error uploading file: ${error}`, 1);
    });
    
    // Clear file input
    fileInput.value = '';
});

// Collaboration event listeners
joinCollab.addEventListener('click', joinCollaboration);
leaveCollab.addEventListener('click', leaveCollaboration);
sendMessage.addEventListener('click', sendChatMessage);
chatMessage.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
        sendChatMessage();
    }
});

// Socket event listeners
socket.on('connect', () => {
    addOutput('Connected to server', 0);
});

socket.on('disconnect', () => {
    addOutput('Disconnected from server', 1);
});

socket.on('error', (data) => {
    addOutput(`Error: ${data.message}`, 1);
});

socket.on('command_result', (data) => {
    addOutput(data.output, data.exit_code);
});

socket.on('system_message', (data) => {
    addOutput(data.message, 0);
});

socket.on('user_joined', (data) => {
    addUser(data.username);
    
    const systemMsg = document.createElement('div');
    systemMsg.classList.add('system-message');
    systemMsg.textContent = `${data.username} joined the collaboration`;
    chatMessages.appendChild(systemMsg);
    chatMessages.scrollTop = chatMessages.scrollHeight;
});

socket.on('user_left', (data) => {
    removeUser(data.username);
    
    const systemMsg = document.createElement('div');
    systemMsg.classList.add('system-message');
    systemMsg.textContent = `${data.username} left the collaboration`;
    chatMessages.appendChild(systemMsg);
    chatMessages.scrollTop = chatMessages.scrollHeight;
});

socket.on('collaboration_message', (data) => {
    addChatMessage(data.username, data.message, data.timestamp);
});


// Helper class for command output
class OutputCapture {
    constructor() {
        this.buffer = [];
    }
    
    write(text) {
        this.buffer.push(text);
    }
    
    flush() {}
    
    get_output() {
        return this.buffer.join('');
    }
}
''')


# Command handler for web interface
def web_interface_command(args: str) -> int:
    """
    Web interface command handler.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Web interface commands")
    parser.add_argument("action", choices=["start", "stop", "status"], 
                        help="Action to perform")
    parser.add_argument("--host", default=DEFAULT_HOST, 
                        help=f"Host to bind to (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, 
                        help=f"Port to listen on (default: {DEFAULT_PORT})")
    parser.add_argument("--users", help="Path to users file")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    # Get registry
    registry = sys.modules.get("__main__").registry
    
    # Get web interface instance
    web_interface = registry.get_component("web_interface")
    
    if not web_interface:
        print("Error: Web interface not initialized")
        print("Creating new web interface instance...")
        
        # Create new web interface instance
        web_interface = WebInterface(cmd_args.host, cmd_args.port)
        
        # Set registry and shell environment
        web_interface.set_registry(registry)
        shell_env = registry.get_component("shell")
        if shell_env:
            web_interface.set_shell_env(shell_env)
        
        # Load users if provided
        if cmd_args.users:
            web_interface.load_users(cmd_args.users)
        
        # Register with registry
        registry.register_component("web_interface", web_interface)
    
    # Perform action
    if cmd_args.action == "start":
        print(f"Starting web interface on {cmd_args.host}:{cmd_args.port}...")
        
        # Start web interface in a new thread
        thread = threading.Thread(target=web_interface.start)
        thread.daemon = True
        thread.start()
        
        print("Web interface started in background")
        print("Access it at http://{host}:{port}/".format(
            host=cmd_args.host if cmd_args.host != "0.0.0.0" else "localhost",
            port=cmd_args.port
        ))
        return 0
    
    elif cmd_args.action == "stop":
        print("Stopping web interface...")
        
        if web_interface.stop():
            print("Web interface stopped")
            return 0
        else:
            print("Error stopping web interface")
            return 1
    
    elif cmd_args.action == "status":
        if not hasattr(web_interface, "running") or not web_interface.running:
            print("Web interface is not running")
            return 1
        
        print("Web interface status:")
        print(f"  Running: {web_interface.running}")
        print(f"  Host: {web_interface.host}")
        print(f"  Port: {web_interface.port}")
        print(f"  Uptime: {time.time() - web_interface.start_time:.2f} seconds")
        print(f"  Sessions: {len(web_interface.sessions)}")
        print(f"  Shell instances: {len(web_interface.shell_instances)}")
        
        return 0
    
    return 1


class OutputCapture:
    """Capture stdout output."""
    
    def __init__(self):
        """Initialize the output capture."""
        self.buffer = []
    
    def write(self, text):
        """
        Write to the buffer.
        
        Args:
            text: Text to write
        """
        self.buffer.append(text)
    
    def flush(self):
        """Flush the buffer."""
        pass
    
    def get_output(self):
        """
        Get the captured output.
        
        Returns:
            Captured output as a string
        """
        return ''.join(self.buffer)
