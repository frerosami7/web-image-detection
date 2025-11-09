# main.py

import os
import sys
from utils.config import load_config
from api.app import create_app

def main():
    # Load configuration settings
    config = load_config()

    # Initialize the application
    app = create_app(config)

    # Start the web interface or API service
    port = config.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port)

if __name__ == "__main__":
    main()