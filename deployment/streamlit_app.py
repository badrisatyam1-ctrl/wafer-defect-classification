"""Default Streamlit entrypoint for the wafer classifier website."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deployment.streamlit_app_v2 import main


if __name__ == "__main__":
    main()
