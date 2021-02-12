# environment
PLATFORM = "mac"  # either of mac, windows_x86, windows_x86_64,
# linux_x86, linux_x86_64, linux_x86_headless, linux_x86_64_headless,
ENV_FILE = {
    "mac": "Banana.app",
    "windows_x86": "Banana_Windows_x86/Banana.exe",
    "windows_x86_64": "Banana_Windows_x86_64/Banana.exe",
    "linux_x86": "Banana_Linux/Banana.x86",
    "linux_x86_64": "/Banana_Linux/Banana.x86_64",
    "linux_x86_headless": "/Banana_Linux_NoVis/Banana.x86",
    "linux_x86_64_headless": "/Banana_Linux_NoVis/Banana.x86_64"
}[PLATFORM]
ENV_PATH = f"./banana_env/{ENV_FILE}"
NUM_OBS = 37
NUM_ACT = 4
TARGET_SCORE = 13

# dqn agent
BUFFER_SIZE = 100
BATCH_SIZE = 32
LR = 5e-4
GAMMA = 0.99
TAU = 1e-2
EPS_DECAY_STEP = 1e-1
HIDDEN_DIM = [64, 128]


