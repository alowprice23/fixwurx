#!/usr/bin/env python3
"""
Remote Shell Access Module

This module provides secure remote access to the FixWurx shell environment 
over SSH and other secure protocols.
"""

import os
import sys
import socket
import threading
import argparse
import logging
import json
import ssl
import time
import base64
import hashlib
import uuid
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("remote_shell.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RemoteShell")

class RemoteShellServer:
    """Server for remote shell access."""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 8022):
        """
        Initialize the remote shell server.
        
        Args:
            host: Host to bind to
            port: Port to listen on
        """
        self.host = host
        self.port = port
        self.server_socket = None
        self.clients = {}
        self.running = False
        self.registry = None
        self.shell_env = None
        self.ssl_context = None
        
        # Authentication
        self.auth_tokens = {}
        self.session_tokens = {}
        
        # Rate limiting
        self.rate_limits = {
            'login_attempts': {
                'max_attempts': 5,
                'time_window': 300,  # 5 minutes
                'clients': {}
            }
        }
    
    def set_registry(self, registry):
        """Set the component registry."""
        self.registry = registry
    
    def set_shell_env(self, shell_env):
        """Set the shell environment."""
        self.shell_env = shell_env
    
    def configure_ssl(self, cert_file: str, key_file: str):
        """
        Configure SSL for secure connections.
        
        Args:
            cert_file: SSL certificate file
            key_file: SSL key file
        """
        self.ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        self.ssl_context.load_cert_chain(certfile=cert_file, keyfile=key_file)
        logger.info(f"SSL configured with cert {cert_file} and key {key_file}")
    
    def start(self):
        """Start the remote shell server."""
        if self.running:
            logger.warning("Server already running")
            return False
        
        try:
            # Create server socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            
            # Wrap with SSL if configured
            if self.ssl_context:
                self.server_socket = self.ssl_context.wrap_socket(self.server_socket, server_side=True)
            
            self.running = True
            logger.info(f"Server started on {self.host}:{self.port}")
            
            # Start listener thread
            self.listener_thread = threading.Thread(target=self._listen_for_clients)
            self.listener_thread.daemon = True
            self.listener_thread.start()
            
            return True
        except Exception as e:
            logger.error(f"Error starting server: {e}")
            return False
    
    def stop(self):
        """Stop the remote shell server."""
        if not self.running:
            logger.warning("Server not running")
            return False
        
        try:
            self.running = False
            
            # Close all client connections
            for client_id, client_info in list(self.clients.items()):
                try:
                    client_info['socket'].close()
                except Exception as e:
                    logger.error(f"Error closing client socket {client_id}: {e}")
            
            # Close server socket
            if self.server_socket:
                self.server_socket.close()
            
            logger.info("Server stopped")
            return True
        except Exception as e:
            logger.error(f"Error stopping server: {e}")
            return False
    
    def _listen_for_clients(self):
        """Listen for client connections."""
        while self.running:
            try:
                # Accept client connection
                client_socket, client_address = self.server_socket.accept()
                logger.info(f"New connection from {client_address}")
                
                # Start client handler thread
                client_id = str(uuid.uuid4())
                client_thread = threading.Thread(
                    target=self._handle_client, 
                    args=(client_socket, client_address, client_id)
                )
                client_thread.daemon = True
                client_thread.start()
                
                # Add to clients dictionary
                self.clients[client_id] = {
                    'socket': client_socket,
                    'address': client_address,
                    'thread': client_thread,
                    'authenticated': False,
                    'username': None,
                    'connected_at': time.time(),
                    'last_activity': time.time()
                }
            except Exception as e:
                if self.running:  # Only log if still supposed to be running
                    logger.error(f"Error accepting client connection: {e}")
                    time.sleep(1)  # Prevent CPU spike on repeated errors
    
    def _handle_client(self, client_socket, client_address, client_id):
        """
        Handle a client connection.
        
        Args:
            client_socket: Client socket
            client_address: Client address (host, port)
            client_id: Client ID
        """
        try:
            # Create client shell
            client_shell = RemoteClientShell(self, client_id)
            
            # Send welcome message
            self._send_message(client_socket, {
                'type': 'welcome',
                'message': 'Welcome to FixWurx Remote Shell',
                'server_time': time.time(),
                'require_auth': True
            })
            
            # Main client loop
            while self.running and client_id in self.clients:
                # Receive message
                message = self._receive_message(client_socket)
                if not message:
                    break
                
                # Update last activity time
                self.clients[client_id]['last_activity'] = time.time()
                
                # Process message
                if message.get('type') == 'auth':
                    self._handle_auth(client_socket, client_id, message)
                elif message.get('type') == 'command':
                    if not self.clients[client_id]['authenticated']:
                        self._send_message(client_socket, {
                            'type': 'error',
                            'message': 'Authentication required'
                        })
                        continue
                    
                    self._handle_command(client_socket, client_id, message, client_shell)
                elif message.get('type') == 'ping':
                    self._send_message(client_socket, {
                        'type': 'pong',
                        'server_time': time.time()
                    })
                else:
                    self._send_message(client_socket, {
                        'type': 'error',
                        'message': 'Unknown message type'
                    })
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        finally:
            # Clean up client
            try:
                client_socket.close()
            except Exception:
                pass
            
            if client_id in self.clients:
                del self.clients[client_id]
            
            logger.info(f"Client {client_id} disconnected")
    
    def _handle_auth(self, client_socket, client_id, message):
        """
        Handle authentication message.
        
        Args:
            client_socket: Client socket
            client_id: Client ID
            message: Authentication message
        """
        # Check rate limiting
        client_ip = self.clients[client_id]['address'][0]
        rate_limit_info = self.rate_limits['login_attempts']
        
        if client_ip not in rate_limit_info['clients']:
            rate_limit_info['clients'][client_ip] = {
                'attempts': 0,
                'first_attempt': time.time()
            }
        
        # Check if rate limited
        client_rate = rate_limit_info['clients'][client_ip]
        time_diff = time.time() - client_rate['first_attempt']
        
        if time_diff > rate_limit_info['time_window']:
            # Reset if outside time window
            client_rate['attempts'] = 0
            client_rate['first_attempt'] = time.time()
        elif client_rate['attempts'] >= rate_limit_info['max_attempts']:
            # Rate limited
            self._send_message(client_socket, {
                'type': 'error',
                'message': 'Too many login attempts, please try again later'
            })
            return
        
        # Increment attempt counter
        client_rate['attempts'] += 1
        
        # Process authentication
        auth_type = message.get('auth_type', 'password')
        
        if auth_type == 'password':
            username = message.get('username')
            password = message.get('password')
            
            if not username or not password:
                self._send_message(client_socket, {
                    'type': 'error',
                    'message': 'Username and password required'
                })
                return
            
            # Authenticate user (example implementation)
            if self._authenticate_user(username, password):
                # Generate session token
                session_token = self._generate_session_token(username)
                
                # Update client info
                self.clients[client_id]['authenticated'] = True
                self.clients[client_id]['username'] = username
                
                # Send success response
                self._send_message(client_socket, {
                    'type': 'auth_success',
                    'username': username,
                    'session_token': session_token
                })
                
                logger.info(f"Client {client_id} authenticated as {username}")
                
                # Reset rate limit counter
                client_rate['attempts'] = 0
            else:
                self._send_message(client_socket, {
                    'type': 'auth_failure',
                    'message': 'Invalid username or password'
                })
                
                logger.warning(f"Failed authentication attempt for user {username} from {client_ip}")
        elif auth_type == 'token':
            token = message.get('token')
            
            if not token:
                self._send_message(client_socket, {
                    'type': 'error',
                    'message': 'Token required'
                })
                return
            
            # Validate token
            username = self._validate_session_token(token)
            
            if username:
                # Update client info
                self.clients[client_id]['authenticated'] = True
                self.clients[client_id]['username'] = username
                
                # Send success response
                self._send_message(client_socket, {
                    'type': 'auth_success',
                    'username': username,
                    'session_token': token
                })
                
                logger.info(f"Client {client_id} authenticated with token as {username}")
                
                # Reset rate limit counter
                client_rate['attempts'] = 0
            else:
                self._send_message(client_socket, {
                    'type': 'auth_failure',
                    'message': 'Invalid or expired token'
                })
        else:
            self._send_message(client_socket, {
                'type': 'error',
                'message': f'Unsupported authentication type: {auth_type}'
            })
    
    def _handle_command(self, client_socket, client_id, message, client_shell):
        """
        Handle command message.
        
        Args:
            client_socket: Client socket
            client_id: Client ID
            message: Command message
            client_shell: Client shell instance
        """
        command = message.get('command')
        
        if not command:
            self._send_message(client_socket, {
                'type': 'error',
                'message': 'Command required'
            })
            return
        
        # Execute command
        try:
            output, exit_code = client_shell.execute_command(command)
            
            # Send response
            self._send_message(client_socket, {
                'type': 'command_result',
                'command': command,
                'output': output,
                'exit_code': exit_code
            })
            
            logger.info(f"Client {client_id} executed command: {command}")
        except Exception as e:
            self._send_message(client_socket, {
                'type': 'error',
                'message': f'Error executing command: {e}'
            })
            
            logger.error(f"Error executing command for client {client_id}: {e}")
    
    def _send_message(self, client_socket, message):
        """
        Send a message to a client.
        
        Args:
            client_socket: Client socket
            message: Message to send
        """
        try:
            # Convert message to JSON
            message_json = json.dumps(message)
            
            # Send message length (4 bytes) followed by message
            message_bytes = message_json.encode('utf-8')
            length_bytes = len(message_bytes).to_bytes(4, byteorder='big')
            
            client_socket.sendall(length_bytes + message_bytes)
            return True
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
    
    def _receive_message(self, client_socket):
        """
        Receive a message from a client.
        
        Args:
            client_socket: Client socket
            
        Returns:
            Received message or None if error
        """
        try:
            # Receive message length (4 bytes)
            length_bytes = client_socket.recv(4)
            if not length_bytes or len(length_bytes) < 4:
                return None
            
            message_length = int.from_bytes(length_bytes, byteorder='big')
            
            # Receive message
            message_bytes = b''
            bytes_received = 0
            
            while bytes_received < message_length:
                chunk = client_socket.recv(min(message_length - bytes_received, 4096))
                if not chunk:
                    return None
                
                message_bytes += chunk
                bytes_received += len(chunk)
            
            # Parse message
            message_json = message_bytes.decode('utf-8')
            message = json.loads(message_json)
            
            return message
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            return None
    
    def _authenticate_user(self, username, password):
        """
        Authenticate a user.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            True if authenticated, False otherwise
        """
        # This is a simple example implementation
        # In a real system, you would check against a secure user database
        
        # Load users from file
        users_file = Path('config/remote_users.json')
        
        if not users_file.exists():
            logger.warning(f"Users file {users_file} not found")
            return False
        
        try:
            with open(users_file, 'r') as f:
                users = json.load(f)
            
            # Check if user exists
            if username not in users:
                return False
            
            # Get user info
            user_info = users[username]
            
            # Check password hash
            password_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()
            
            return password_hash == user_info.get('password_hash')
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return False
    
    def _generate_session_token(self, username):
        """
        Generate a session token for a user.
        
        Args:
            username: Username
            
        Returns:
            Session token
        """
        token = str(uuid.uuid4())
        expires = time.time() + 86400  # 24 hours
        
        self.session_tokens[token] = {
            'username': username,
            'created_at': time.time(),
            'expires': expires
        }
        
        return token
    
    def _validate_session_token(self, token):
        """
        Validate a session token.
        
        Args:
            token: Session token
            
        Returns:
            Username if valid, None otherwise
        """
        if token not in self.session_tokens:
            return None
        
        token_info = self.session_tokens[token]
        
        # Check if expired
        if time.time() > token_info['expires']:
            del self.session_tokens[token]
            return None
        
        return token_info['username']
    
    def broadcast_message(self, message):
        """
        Broadcast a message to all connected clients.
        
        Args:
            message: Message to broadcast
            
        Returns:
            Number of clients message was sent to
        """
        sent_count = 0
        
        for client_id, client_info in list(self.clients.items()):
            if client_info['authenticated']:
                if self._send_message(client_info['socket'], message):
                    sent_count += 1
        
        return sent_count
    
    def get_status(self):
        """
        Get server status.
        
        Returns:
            Server status information
        """
        return {
            'running': self.running,
            'host': self.host,
            'port': self.port,
            'ssl_enabled': self.ssl_context is not None,
            'client_count': len(self.clients),
            'clients': [
                {
                    'id': client_id,
                    'address': client_info['address'],
                    'authenticated': client_info['authenticated'],
                    'username': client_info['username'],
                    'connected_at': client_info['connected_at'],
                    'last_activity': client_info['last_activity']
                }
                for client_id, client_info in self.clients.items()
            ]
        }


class RemoteClientShell:
    """Shell for remote clients."""
    
    def __init__(self, server, client_id):
        """
        Initialize the remote client shell.
        
        Args:
            server: Remote shell server
            client_id: Client ID
        """
        self.server = server
        self.client_id = client_id
        self.registry = server.registry
        self.shell_env = server.shell_env
        self.cwd = os.getcwd()
        self.environment = {
            'PATH': os.environ.get('PATH', ''),
            'HOME': os.environ.get('HOME', ''),
            'USER': server.clients[client_id].get('username', 'unknown')
        }
        self.history = []
    
    def execute_command(self, command):
        """
        Execute a shell command.
        
        Args:
            command: Command to execute
            
        Returns:
            Tuple of (output, exit_code)
        """
        # Add to history
        self.history.append(command)
        
        # Check if shell environment is available
        if not self.shell_env:
            return "Error: Shell environment not available", 1
        
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
                return self._change_directory(args)
            
            # Find command handler
            handler_info = self.registry.get_command_handler(cmd)
            
            if handler_info:
                # Redirect stdout to capture output
                original_stdout = sys.stdout
                sys.stdout = OutputCapture()
                
                try:
                    # Execute command handler
                    handler = handler_info["handler"]
                    exit_code = handler(args)
                    
                    # Get captured output
                    output = sys.stdout.get_output()
                    
                    return output, exit_code
                finally:
                    # Restore stdout
                    sys.stdout = original_stdout
            else:
                return f"Unknown command: {cmd}", 1
        except Exception as e:
            return f"Error executing command: {e}", 1
    
    def _change_directory(self, path):
        """
        Change working directory.
        
        Args:
            path: Directory path
            
        Returns:
            Tuple of (output, exit_code)
        """
        try:
            # Handle special cases
            if not path:
                path = os.environ.get('HOME', '/')
            elif path == '-':
                # Go to previous directory
                path = getattr(self, 'prev_cwd', self.cwd)
            
            # Expand path
            if path.startswith('~'):
                path = os.path.expanduser(path)
            elif not os.path.isabs(path):
                path = os.path.join(self.cwd, path)
            
            # Normalize path
            path = os.path.normpath(path)
            
            # Check if directory exists
            if not os.path.isdir(path):
                return f"cd: {path}: No such directory", 1
            
            # Save previous directory
            self.prev_cwd = self.cwd
            
            # Change directory
            self.cwd = path
            os.chdir(path)
            
            return "", 0
        except Exception as e:
            return f"cd: {e}", 1


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


class RemoteShellClient:
    """Client for remote shell access."""
    
    def __init__(self, host: str = 'localhost', port: int = 8022):
        """
        Initialize the remote shell client.
        
        Args:
            host: Remote host
            port: Remote port
        """
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        self.authenticated = False
        self.username = None
        self.session_token = None
        self.ssl_context = None
    
    def configure_ssl(self, cert_file: str = None):
        """
        Configure SSL for secure connections.
        
        Args:
            cert_file: SSL certificate file (optional, for certificate verification)
        """
        self.ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        
        if cert_file:
            self.ssl_context.load_verify_locations(cert_file)
        else:
            # This is insecure - only use for testing
            self.ssl_context.check_hostname = False
            self.ssl_context.verify_mode = ssl.CERT_NONE
    
    def connect(self):
        """
        Connect to the remote shell server.
        
        Returns:
            True if connected, False otherwise
        """
        if self.connected:
            logger.warning("Already connected")
            return True
        
        try:
            # Create socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            # Wrap with SSL if configured
            if self.ssl_context:
                self.socket = self.ssl_context.wrap_socket(self.socket, server_hostname=self.host)
            
            # Connect to server
            self.socket.connect((self.host, self.port))
            self.connected = True
            
            # Receive welcome message
            welcome = self._receive_message()
            
            if not welcome or welcome.get('type') != 'welcome':
                logger.error("Invalid welcome message")
                self.disconnect()
                return False
            
            logger.info(f"Connected to {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Error connecting to {self.host}:{self.port}: {e}")
            self.disconnect()
            return False
    
    def disconnect(self):
        """Disconnect from the remote shell server."""
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
            
            self.socket = None
        
        self.connected = False
        self.authenticated = False
        logger.info("Disconnected")
    
    def authenticate(self, username: str, password: str):
        """
        Authenticate with the remote shell server.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            True if authenticated, False otherwise
        """
        if not self.connected:
            logger.error("Not connected")
            return False
        
        if self.authenticated:
            logger.warning("Already authenticated")
            return True
        
        try:
            # Send authentication message
            self._send_message({
                'type': 'auth',
                'auth_type': 'password',
                'username': username,
                'password': password
            })
            
            # Receive response
            response = self._receive_message()
            
            if not response:
                logger.error("No response received")
                return False
            
            if response.get('type') == 'auth_success':
                self.authenticated = True
                self.username = response.get('username', username)
                self.session_token = response.get('session_token')
                
                logger.info(f"Authenticated as {self.username}")
                return True
            else:
                logger.error(f"Authentication failed: {response.get('message', 'Unknown error')}")
                return False
        except Exception as e:
            logger.error(f"Error authenticating: {e}")
            return False
    
    def authenticate_with_token(self, token: str):
        """
        Authenticate with the remote shell server using a token.
        
        Args:
            token: Authentication token
            
        Returns:
            True if authenticated, False otherwise
        """
        if not self.connected:
            logger.error("Not connected")
            return False
        
        if self.authenticated:
            logger.warning("Already authenticated")
            return True
        
        try:
            # Send authentication message
            self._send_message({
                'type': 'auth',
                'auth_type': 'token',
                'token': token
            })
            
            # Receive response
            response = self._receive_message()
            
            if not response:
                logger.error("No response received")
                return False
            
            if response.get('type') == 'auth_success':
                self.authenticated = True
                self.username = response.get('username')
                self.session_token = response.get('session_token')
                
                logger.info(f"Authenticated as {self.username}")
                return True
            else:
                logger.error(f"Authentication failed: {response.get('message', 'Unknown error')}")
                return False
        except Exception as e:
            logger.error(f"Error authenticating: {e}")
            return False
    
    def execute_command(self, command: str):
        """
        Execute a command on the remote shell server.
        
        Args:
            command: Command to execute
            
        Returns:
            Tuple of (output, exit_code) or (None, None) if error
        """
        if not self.connected:
            logger.error("Not connected")
            return None, None
        
        if not self.authenticated:
            logger.error("Not authenticated")
            return None, None
        
        try:
            # Send command message
            self._send_message({
                'type': 'command',
                'command': command
            })
            
            # Receive response
            response = self._receive_message()
            
            if not response:
                logger.error("No response received")
                return None, None
            
            if response.get('type') == 'command_result':
                return response.get('output', ''), response.get('exit_code', 1)
            else:
                logger.error(f"Error executing command: {response.get('message', 'Unknown error')}")
                return None, None
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            return None, None
    
    def ping(self):
        """
        Ping the remote shell server.
        
        Returns:
            Round-trip time in seconds or None if error
        """
        if not self.connected:
            logger.error("Not connected")
            return None
        
        try:
            # Send ping message
            start_time = time.time()
            
            self._send_message({
                'type': 'ping',
                'client_time': start_time
            })
            
            # Receive pong message
            response = self._receive_message()
            
            if not response or response.get('type') != 'pong':
                logger.error("Invalid ping response")
                return None
            
            end_time = time.time()
            
            # Calculate round-trip time
            rtt = end_time - start_time
            
            return rtt
        except Exception as e:
            logger.error(f"Error pinging server: {e}")
            return None
    
    def _send_message(self, message):
        """
        Send a message to the server.
        
        Args:
            message: Message to send
            
        Returns:
            True if sent, False otherwise
        """
        try:
            # Convert message to JSON
            message_json = json.dumps(message)
            
            # Send message length (4 bytes) followed by message
            message_bytes = message_json.encode('utf-8')
            length_bytes = len(message_bytes).to_bytes(4, byteorder='big')
            
            self.socket.sendall(length_bytes + message_bytes)
            return True
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
    
    def _receive_message(self):
        """
        Receive a message from the server.
        
        Returns:
            Received message or None if error
        """
        try:
            # Receive message length (4 bytes)
            length_bytes = self.socket.recv(4)
            if not length_bytes or len(length_bytes) < 4:
                return None
            
            message_length = int.from_bytes(length_bytes, byteorder='big')
            
            # Receive message
            message_bytes = b''
            bytes_received = 0
            
            while bytes_received < message_length:
                chunk = self.socket.recv(min(message_length - bytes_received, 4096))
                if not chunk:
                    return None
                
                message_bytes += chunk
                bytes_received += len(chunk)
            
            # Parse message
            message_json = message_bytes.decode('utf-8')
            message = json.loads(message_json)
            
            return message
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            return None


# Command handler for remote shell
def remote_shell_command(args: str) -> int:
    """
    Remote shell command.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Remote shell commands")
    parser.add_argument("action", choices=["start", "stop", "status", "client"], 
                        help="Action to perform")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (server) or connect to (client)")
    parser.add_argument("--port", type=int, default=8022, help="Port to listen on (server) or connect to (client)")
    parser.add_argument("--ssl-cert", help="SSL certificate file")
    parser.add_argument("--ssl-key", help="SSL key file")
    parser.add_argument("--username", help="Username for client authentication")
    parser.add_argument("--password", help="Password for client authentication")
    parser.add_argument("--token", help="Token for client authentication")
    parser.add_argument("--command", help="Command to execute (client)")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    # Get registry
    registry = sys.modules.get("__main__").registry
    
    # Get remote shell server instance
    remote_shell_server = registry.get_component("remote_shell_server")
    
    # Handle server actions
    if cmd_args.action in ["start", "stop", "status"]:
        if not remote_shell_server:
            print("Error: Remote shell server not initialized")
            print("Creating new server instance...")
            
            # Create new server instance
            remote_shell_server = RemoteShellServer(cmd_args.host, cmd_args.port)
            
            # Set registry and shell environment
            remote_shell_server.set_registry(registry)
            shell_env = registry.get_component("shell")
            if shell_env:
                remote_shell_server.set_shell_env(shell_env)
            
            # Register with registry
            registry.register_component("remote_shell_server", remote_shell_server)
        
        if cmd_args.action == "start":
            # Configure SSL if provided
            if cmd_args.ssl_cert and cmd_args.ssl_key:
                remote_shell_server.configure_ssl(cmd_args.ssl_cert, cmd_args.ssl_key)
            
            # Start server
            if remote_shell_server.start():
                print(f"Remote shell server started on {cmd_args.host}:{cmd_args.port}")
                return 0
            else:
                print("Error starting remote shell server")
                return 1
        
        elif cmd_args.action == "stop":
            # Stop server
            if remote_shell_server.stop():
                print("Remote shell server stopped")
                return 0
            else:
                print("Error stopping remote shell server")
                return 1
        
        elif cmd_args.action == "status":
            # Get server status
            status = remote_shell_server.get_status()
            
            print("Remote Shell Server Status:")
            print(f"Running: {status['running']}")
            print(f"Host: {status['host']}")
            print(f"Port: {status['port']}")
            print(f"SSL Enabled: {status['ssl_enabled']}")
            print(f"Connected Clients: {status['client_count']}")
            
            if status['client_count'] > 0:
                print("\nConnected Clients:")
                for client in status['clients']:
                    print(f"  ID: {client['id']}")
                    print(f"    Address: {client['address'][0]}:{client['address'][1]}")
                    print(f"    Authenticated: {client['authenticated']}")
                    if client['authenticated']:
                        print(f"    Username: {client['username']}")
                    print(f"    Connected At: {time.ctime(client['connected_at'])}")
                    print(f"    Last Activity: {time.ctime(client['last_activity'])}")
            
            return 0
    
    # Handle client action
    elif cmd_args.action == "client":
        # Create client
        client = RemoteShellClient(cmd_args.host, cmd_args.port)
        
        # Configure SSL if provided
        if cmd_args.ssl_cert:
            client.configure_ssl(cmd_args.ssl_cert)
        
        # Connect to server
        print(f"Connecting to {cmd_args.host}:{cmd_args.port}...")
        if not client.connect():
            print("Error connecting to server")
            return 1
        
        print("Connected to server")
        
        # Authenticate
        if cmd_args.token:
            print("Authenticating with token...")
            if not client.authenticate_with_token(cmd_args.token):
                print("Authentication failed")
                client.disconnect()
                return 1
        elif cmd_args.username and cmd_args.password:
            print(f"Authenticating as {cmd_args.username}...")
            if not client.authenticate(cmd_args.username, cmd_args.password):
                print("Authentication failed")
                client.disconnect()
                return 1
        else:
            print("Error: Authentication required (--token or --username and --password)")
            client.disconnect()
            return 1
        
        print("Authentication successful")
        
        # Execute command if provided
        if cmd_args.command:
            print(f"Executing command: {cmd_args.command}")
            output, exit_code = client.execute_command(cmd_args.command)
            
            if output is not None:
                print("\nCommand Output:")
                print(output)
                print(f"\nExit code: {exit_code}")
                
                client.disconnect()
                return exit_code
            else:
                print("Error executing command")
                client.disconnect()
                return 1
        else:
            # Interactive mode
            print("Starting interactive mode (type 'exit' to quit)")
            print("Session token:", client.session_token)
            
            try:
                while True:
                    command = input("\n> ")
                    
                    if command.lower() in ['exit', 'quit']:
                        break
                    
                    output, exit_code = client.execute_command(command)
                    
                    if output is not None:
                        print(output)
                        if exit_code != 0:
                            print(f"Exit code: {exit_code}")
                    else:
                        print("Error executing command")
            except KeyboardInterrupt:
                print("\nInterrupted")
            except EOFError:
                print("\nEnd of input")
            finally:
                client.disconnect()
                print("Disconnected from server")
            
            return 0
    
    return 1
