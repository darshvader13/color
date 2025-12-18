import requests
import base64

url = "https://darshvader13--deoldify-colorization-colorize-endpoint.modal.run"

file_path = r"C:\Users\dbala\Desktop\color\bw_resized\hunter x hunter\6\55.jpg"

with open(file_path, "rb") as f:
    files = {"file": f}
    res = requests.post(url, files=files)

data = res.json()
print(data)

img_bytes = base64.b64decode(data["image_base64"])
with open("output.png", "wb") as f:
    f.write(img_bytes)

print("Saved output.png")