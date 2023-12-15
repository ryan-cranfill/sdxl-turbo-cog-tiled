import time
import requests
import cv2
import base64
import numpy as np

SERVER_URL = 'http://192.168.1.17:5001'

input_dict = {
    "input": {
        "prompt": "spaghetti, oil on canvas",
        "prompt_suffixes": "red colored; blue colored; green colored; orange colored",
        "negative_prompt": "blurry, bad, low quality, low resolution",
        "num_outputs": 4,
        "num_inference_steps": 4,
        "width": 512,
        "height": 512,
        "final_width": 512,
        "final_height": 512,
    }
}

start = time.time()

response = requests.post(SERVER_URL + '/predictions', json=input_dict)
response_time = time.time() - start

if response.ok:
    # Output images are base64 encoded pngs in response.json()['output']
    data = response.json()
    for i, image in enumerate(data['output']):
        tile_x = 2
        tile_y = 2
        # Remove the header and decode the image
        image = image.split(',')[1]
        image = base64.b64decode(image)
        # Convert to numpy array
        image = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
        # Tile the image
        image = np.tile(image, (tile_x, tile_y, 1))
        # Save the image
        cv2.imwrite(f'output_{i}.png', image)

end = time.time()
print(f'Time taken: {end - start}')
print(f'Response time: {response_time}')


        # with open(f'output_{i}.png', 'wb') as f:
        #     f.write(base64.b64decode(image))
