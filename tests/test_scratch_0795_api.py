import urllib.request
import json
import os

filepath = 'dataset/train/scratch/augmented_scratch_0795.png'
with open(filepath, 'rb') as f:
    file_bytes = f.read()

boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
body = (
    f'--{boundary}\r\n'
    f'Content-Disposition: form-data; name="file"; filename="{os.path.basename(filepath)}"\r\n'
    f'Content-Type: image/png\r\n\r\n'
).encode('utf-8') + file_bytes + f'\r\n--{boundary}--\r\n'.encode('utf-8')

req = urllib.request.Request(
    'http://127.0.0.1:8000/api/predict',
    data=body,
    headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
)
resp = urllib.request.urlopen(req)
res = json.loads(resp.read().decode())
print('Live Prediction on augmented_scratch_0795.png:')
print('Status:', res.get('status'))
print('Class:', res.get('class'))
print('Confidence:', res.get('confidence'))
print('Has overlay_b64:', bool(res.get('overlay_b64')))
print('Has heatmap_raw_b64:', bool(res.get('heatmap_raw_b64')))
