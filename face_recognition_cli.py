#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Command-line Interface for Unified Face Recognition System
This script provides a simpler command-line interface for accessing the
main functionality of the face recognition and authentication system.
"""

import os
import argparse
import subprocess
import sys

# Fix OpenMP conflict issue early
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def main():
    """
    Command-line interface for the Face Recognition system.
    """
    parser = argparse.ArgumentParser(description='Face Recognition and Authentication CLI')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Register command
    register_parser = subparsers.add_parser('register', help='Register a new user')
    register_parser.add_argument('username', help='Username to register')
    
    # Recognize command
    recognize_parser = subparsers.add_parser('recognize', help='Start face recognition')
    recognize_parser.add_argument('--threshold', type=float, default=0.65,
                                help='Recognition threshold (0.0-1.0)')
    
    # Authenticate command
    auth_parser = subparsers.add_parser('authenticate', help='Start facial authentication')
    auth_parser.add_argument('--mode', type=str, default='single', choices=['single', 'continuous'],
                           help='Authentication mode: single (one-time) or continuous')
    auth_parser.add_argument('--duration', type=int, default=30,
                           help='Duration for continuous mode in seconds (0 for infinite)')
    auth_parser.add_argument('--liveness', action='store_true',
                           help='Enable liveness detection')
    auth_parser.add_argument('--threshold', type=float, default=0.65,
                           help='Recognition threshold (0.0-1.0)')
    auth_parser.add_argument('--monitor', action='store_true',
                           help='Enable system performance monitoring')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup project structure')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Remove redundant files after consolidation')
    
    args = parser.parse_args()
    
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.command == 'register':
        # Run the registration mode with the provided username
        cmd = [sys.executable, os.path.join(script_dir, 'main.py'), 
               '--mode', 'registration', '--username', args.username]
        subprocess.run(cmd)
    
    elif args.command == 'recognize':
        # Run the recognition mode
        cmd = [sys.executable, os.path.join(script_dir, 'main.py'), 
               '--mode', 'recognition', '--threshold', str(args.threshold)]
        subprocess.run(cmd)
    
    elif args.command == 'authenticate':
        # Run the authentication mode with the appropriate options
        cmd = [sys.executable, os.path.join(script_dir, 'main.py'), 
               '--mode', 'authentication', 
               '--auth-mode', args.mode,
               '--threshold', str(args.threshold)]
        
        if args.mode == 'continuous':
            cmd.extend(['--duration', str(args.duration)])
        
        if args.liveness:
            cmd.append('--liveness')
            
        if args.monitor:
            cmd.append('--monitor')
            
        subprocess.run(cmd)
    
    elif args.command == 'setup':
        # Run the setup script
        cmd = [sys.executable, os.path.join(script_dir, 'setup_project.py')]
        subprocess.run(cmd)
    
    elif args.command == 'cleanup':
        # Run the cleanup script
        cmd = [sys.executable, os.path.join(script_dir, 'cleanup.py')]
        subprocess.run(cmd)
    
    else:
        # If no command is provided, show help
        parser.print_help()

if __name__ == '__main__':
    main()
