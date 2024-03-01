from app import app
from dotenv import load_dotenv
import os
# Loading environment variables from .env file
load_dotenv()
# Accessing environment variables
host_no = os.getenv("host")
port_no = os.getenv("port")
debug_mode = os.getenv("DEBUG")


if __name__ == '__main__':
    app.run(debug=debug_mode, host=host_no, port=port_no)