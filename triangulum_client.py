#!/usr/bin/env python3
"""
Triangulum Client Module - Fixed Version

This module provides the TriangulumClient class which is used by other Triangulum components.
"""

import os
import sys
import json
import logging
import time
import threading
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("triangulum_client.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("TriangulumClient")

# Mock for testing
MOCK_MODE = os.environ.get("TRIANGULUM_TEST_MODE", "0") == "1"
if MOCK_MODE:
    # In mock mode, we'll create a mock class
    class MockRequests:
        @staticmethod
        def get(url, headers=None, **kwargs):
            if "/ping" in url:
                return MockResponse(200, {"status": "ok"})
            return MockResponse(404, {"status": "error", "message": "Not found"})
        
        @staticmethod
        def post(url, headers=None, json=None, **kwargs):
            if "/heartbeat" in url:
                return MockResponse(200, {"status": "ok"})
            return MockResponse(200, {"status": "ok", "data": json})
        
        put = post
        delete = get
    
    class MockResponse:
        def __init__(self, status_code=200, json_data=None, text=""):
            self.status_code = status_code
            self._json_data = json_data or {}
            self.text = text
        
        def json(self):
            return self._json_data
    
    requests = MockRequests()
else:
    # In normal mode, we'll use the real requests module
    try:
        import requests
    except ImportError:
        logger.error("Could not import requests module")
        requests = None

class TriangulumClient:
    """
    Client for communicating with Triangulum.
    """
    
    # Static variables
    is_connected_val = False
    last_heartbeat = None
    api_calls = 0
    api_errors = 0
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Triangulum client.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        self.base_url = self.config.get("triangulum_url", "http://localhost:8081")
        self.api_key = self.config.get("triangulum_api_key", "")
        self.is_connected_val = False
        self.heartbeat_thread = None
        self.heartbeat_interval = self.config.get("heartbeat_interval", 30)  # seconds
        self.stop_event = threading.Event()
        
        logger.info("Triangulum client initialized")
    
    def connect(self) -> bool:
        """
        Connect to Triangulum.
        
        Returns:
            Whether the connection was successful
        """
        try:
            if MOCK_MODE:
                # In mock mode, just simulate a successful connection
                self.is_connected_val = True
                TriangulumClient.is_connected_val = True
                TriangulumClient.last_heartbeat = time.time()
                
                # Start heartbeat thread
                self.stop_event.clear()
                self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop)
                self.heartbeat_thread.daemon = True
                self.heartbeat_thread.start()
                
                logger.info("Connected to Triangulum (mock mode)")
                return True
            else:
                # Test connection
                response = self._api_request("GET", "/ping")
                
                if response.get("status") == "ok":
                    self.is_connected_val = True
                    TriangulumClient.is_connected_val = True
                    TriangulumClient.last_heartbeat = time.time()
                    
                    # Start heartbeat thread
                    self.stop_event.clear()
                    self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop)
                    self.heartbeat_thread.daemon = True
                    self.heartbeat_thread.start()
                    
                    logger.info("Connected to Triangulum")
                    return True
                else:
                    logger.error(f"Failed to connect to Triangulum: {response.get('error', 'Unknown error')}")
                    return False
        except Exception as e:
            logger.error(f"Error connecting to Triangulum: {e}")
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from Triangulum.
        
        Returns:
            Whether the disconnection was successful
        """
        self.is_connected_val = False
        TriangulumClient.is_connected_val = False
        
        # Stop heartbeat thread
        self.stop_event.set()
        
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=5)
            if self.heartbeat_thread.is_alive():
                logger.warning("Heartbeat thread did not terminate gracefully")
        
        logger.info("Disconnected from Triangulum")
        return True
    
    def _heartbeat_loop(self) -> None:
        """
        Heartbeat loop.
        """
        while not self.stop_event.is_set():
            try:
                self.send_heartbeat()
                
                # Sleep until next heartbeat
                self.stop_event.wait(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                self.stop_event.wait(5)  # Wait a bit before retrying
    
    def send_heartbeat(self) -> bool:
        """
        Send heartbeat to Triangulum.
        
        Returns:
            Whether the heartbeat was successful
        """
        try:
            if MOCK_MODE:
                # In mock mode, just simulate a successful heartbeat
                TriangulumClient.last_heartbeat = time.time()
                logger.debug("Sent heartbeat to Triangulum (mock mode)")
                return True
            else:
                response = self._api_request("POST", "/heartbeat")
                
                if response.get("status") == "ok":
                    TriangulumClient.last_heartbeat = time.time()
                    logger.debug("Sent heartbeat to Triangulum")
                    return True
                else:
                    logger.warning(f"Failed to send heartbeat to Triangulum: {response.get('error', 'Unknown error')}")
                    return False
        except Exception as e:
            logger.error(f"Error sending heartbeat to Triangulum: {e}")
            return False
    
    def _api_request(self, method: str, endpoint: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make an API request to Triangulum.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request data
            
        Returns:
            Response data
        """
        url = f"{self.base_url}{endpoint}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            TriangulumClient.api_calls += 1
            
            if method == "GET":
                response = requests.get(url, headers=headers)
            elif method == "POST":
                response = requests.post(url, headers=headers, json=data)
            elif method == "PUT":
                response = requests.put(url, headers=headers, json=data)
            elif method == "DELETE":
                response = requests.delete(url, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            if response.status_code >= 400:
                TriangulumClient.api_errors += 1
                return {
                    "status": "error",
                    "error": f"HTTP error: {response.status_code}",
                    "response": response.text
                }
            
            return response.json()
        except Exception as e:
            TriangulumClient.api_errors += 1
            logger.error(f"Error making API request to Triangulum: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    @staticmethod
    def is_connected() -> bool:
        """
        Check if connected to Triangulum.
        
        Returns:
            Whether connected to Triangulum
        """
        return TriangulumClient.is_connected_val
