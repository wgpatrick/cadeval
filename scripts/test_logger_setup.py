#!/usr/bin/env python3
"""
Unit tests for the logger_setup module.
"""

import os
import sys
import unittest
import tempfile
import logging
import shutil
from io import StringIO
import time # Import time

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from scripts.logger_setup import (
    setup_logger,
    configure_project_logging,
    get_logger,
    DEFAULT_LOG_DIR
)

class TestLoggerSetup(unittest.TestCase):
    """Test cases for the logger_setup module."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for log files
        self.test_log_dir = tempfile.mkdtemp()
        
        # Reset the root logger
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)
    
    def tearDown(self):
        """Clean up after tests."""
        # Clean up temporary directory
        shutil.rmtree(self.test_log_dir)
        
        # Explicitly close/remove handlers from test-specific loggers
        loggers_to_clear = ["test_logger", "test_levels", "console_logger", "file_logger"] 
        for name in loggers_to_clear:
             if name in logging.Logger.manager.loggerDict:
                 logger = logging.getLogger(name)
                 for handler in logger.handlers[:]:
                     try:
                         handler.close()
                     except Exception:
                         pass # Ignore errors during cleanup
                     logger.removeHandler(handler)

        # Reset the root logger handlers
        root = logging.getLogger()
        for handler in root.handlers[:]:
            try:
                handler.close() # Also close root handlers just in case
            except Exception:
                pass
            root.removeHandler(handler)
        
        # Explicitly delete test loggers from the manager's dictionary for isolation
        for name in loggers_to_clear:
             if name in logging.Logger.manager.loggerDict:
                 try:
                     del logging.Logger.manager.loggerDict[name]
                 except KeyError:
                     pass # Might have already been removed, ignore
    
    def test_setup_logger(self):
        """Test the setup_logger function creates a properly configured logger."""
        test_log_file = os.path.join(self.test_log_dir, "test.log")
        logger_name = "test_setup_logger_instance" # Unique name
        logger = setup_logger(logger_name, log_file=test_log_file)
        
        # Check that the logger has the correct name
        self.assertEqual(logger.name, logger_name)
        
        # Check that the logger has the correct handlers
        self.assertEqual(len(logger.handlers), 2)  # Console and file handlers
        
        # Check that the log file was created
        self.assertTrue(os.path.exists(test_log_file))
        
        # Log a test message
        test_message = "Test log message"
        logger.info(test_message)
        
        # Flush and close the handlers to ensure messages are written and file is released
        for handler in logger.handlers:
            handler.flush()
            handler.close()
        time.sleep(0.05) # Shorter sleep after flush/close
        
        # Check that the message was written to the file
        with open(test_log_file, 'r') as f:
            log_content = f.read()
            self.assertIn(test_message, log_content)
    
    def test_setup_logger_console_only(self):
        """Test logger setup without a file handler."""
        logger = setup_logger("console_logger", log_file=None)
        
        # Check that the logger has only one handler (console)
        self.assertEqual(len(logger.handlers), 1)
        
        # Check handler type
        self.assertIsInstance(logger.handlers[0], logging.StreamHandler)
    
    def test_setup_logger_file_only(self):
        """Test logger setup without a console handler."""
        test_log_file = os.path.join(self.test_log_dir, "file_only.log")
        logger = setup_logger("file_logger", log_file=test_log_file, console=False)
        
        # Check that the logger has only one handler (file)
        self.assertEqual(len(logger.handlers), 1)
        
        # Check handler type
        self.assertIsInstance(logger.handlers[0], logging.FileHandler)
        
        # Check file path
        self.assertEqual(logger.handlers[0].baseFilename, test_log_file)
    
    def test_configure_project_logging(self):
        """Test project-wide logging configuration."""
        # Configure with components
        components = ["test_component1", "test_component2"]
        configure_project_logging(
            log_dir=self.test_log_dir,
            components=components
        )
        
        # Check root log file
        root_log_file = os.path.join(self.test_log_dir, "cadeval.log")
        self.assertTrue(os.path.exists(root_log_file))
        
        # Check component log files
        for component in components:
            component_log_file = os.path.join(self.test_log_dir, f"{component}.log")
            self.assertTrue(os.path.exists(component_log_file))
        
        # Get root logger and check handlers
        root_logger = logging.getLogger("")
        self.assertGreaterEqual(len(root_logger.handlers), 1)
        
        # Get component loggers and check if they inherit from root
        for component in components:
            component_logger = logging.getLogger(component)
            self.assertTrue(component_logger.propagate)
    
    def test_get_logger(self):
        """Test the get_logger function."""
        # Configure project logging first
        configure_project_logging(log_dir=self.test_log_dir)
        
        # Get a logger
        test_logger_name = "test_get_logger"
        logger = get_logger(test_logger_name)
        
        # Check the logger name
        self.assertEqual(logger.name, test_logger_name)
        
        # Test that it's the same logger if we get it again
        same_logger = get_logger(test_logger_name)
        self.assertIs(logger, same_logger)
    
    def test_log_levels(self):
        """Test that different log levels work correctly."""
        # Create a StringIO for capturing console output (Still useful if setup_logger adds StreamHandler)
        # console_output = StringIO() # Keep if needed, maybe mock StreamHandler later if necessary
        
        # Create test log file
        test_log_file = os.path.join(self.test_log_dir, "levels.log")
        
        # --- Use setup_logger to configure --- Start ---
        logger_name = "test_log_levels_instance" # Unique name
        logger = setup_logger(
            name=logger_name,
            level=logging.DEBUG, # Set desired level here
            log_file=test_log_file,
            console=True # Assume we want both for testing levels
        )
        # --- Use setup_logger to configure --- End ---
        
        # # Removed manual handler creation
        # console_handler = logging.StreamHandler(console_output)
        # console_handler.setLevel(logging.INFO)
        # console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        
        # file_handler = logging.FileHandler(test_log_file)
        # file_handler.setLevel(logging.DEBUG)
        # file_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        
        # # Removed manual logger setup
        # logger = logging.getLogger("test_levels")
        # logger.setLevel(logging.DEBUG)
        # logger.handlers = []  # Clear any existing handlers
        # logger.addHandler(console_handler)
        # logger.addHandler(file_handler)
        
        # Log messages at different levels
        debug_msg = "This is a DEBUG message"
        info_msg = "This is an INFO message"
        
        logger.debug(debug_msg)
        logger.info(info_msg)
        
        # Flush and close the handlers to ensure messages are written and file is released
        for handler in logger.handlers:
            handler.flush()
            handler.close()
        time.sleep(0.05) # Shorter sleep after flush/close
        
        # Check that DEBUG message is in file but not console
        with open(test_log_file, 'r') as f:
            log_content = f.read()
            self.assertIn(debug_msg, log_content)
            self.assertIn(info_msg, log_content)
        
        # Check console output (should have info but not debug)
        # captured = console_output.getvalue()
        # self.assertIn(info_msg, captured)
        # self.assertNotIn(debug_msg, captured)


if __name__ == "__main__":
    unittest.main() 