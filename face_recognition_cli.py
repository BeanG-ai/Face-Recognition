import os
import argparse
import subprocess
import sys

def main():
    """
    Command-line interface for the Face Recognition system.
    """
    parser = argparse.ArgumentParser(description='Face Recognition CLI')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Register command
    register_parser = subparsers.add_parser('register', help='Register a new user')
    register_parser.add_argument('username', help='Username to register')
    
    # Recognize command
    recognize_parser = subparsers.add_parser('recognize', help='Start face recognition')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup project structure')
    
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
               '--mode', 'recognition']
        subprocess.run(cmd)
    
    elif args.command == 'setup':
        # Run the setup script
        cmd = [sys.executable, os.path.join(script_dir, 'setup_project.py')]
        subprocess.run(cmd)
    
    else:
        # If no command is provided, show help
        parser.print_help()

if __name__ == '__main__':
    main()
