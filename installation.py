import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("numpy")
install("imutils")
install("cv2")
install("pandas")
install("torch")
install("torchvision")

