import os
import sys

# Put the SDK's src/ first so `import assemblyai.agent` resolves to this package
# even if the real `assemblyai` SDK is installed.
SRC = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
sys.path.insert(0, SRC)
