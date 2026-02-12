import socketio
import eventlet
import numpy as np
from flask import Flask
import base64
from io import BytesIO
from PIL import Image
import cv2
import tensorflow as tf
from collections import deque
import time
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

sio = socketio.Server()
app = Flask(__name__) 

# CONFIGURATION
speed_limit = 50
STEERING_SMOOTHING = 1  # Smooth over last 3 frames
THROTTLE_SMOOTHING = 3
MAX_STEERING_CHANGE = 0.8  # Prevent sudden steering changes

# History buffers
steering_history = deque(maxlen=STEERING_SMOOTHING)
throttle_history = deque(maxlen=THROTTLE_SMOOTHING)
last_steering = 0.0

# Performance tracking
frame_times = deque(maxlen=100)

# OPTIMIZED: Pre-allocate array for faster reshaping
IMAGE_SHAPE = (1, 66, 200, 3)

def img_preprocess(img):
    """Optimized preprocessing with minimal operations"""
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img

def smooth_control(new_value, history):
    """Exponential weighted moving average"""
    history.append(new_value)
    if len(history) == 1:
        return new_value
    
    # More weight on recent values
    weights = np.linspace(0.5, 1.0, len(history))
    weights = weights / weights.sum()
    
    return np.average(list(history), weights=weights)

def clip_steering_change(new_steering, last_steering, max_change=MAX_STEERING_CHANGE):
    """Prevent sudden steering jerks"""
    change = new_steering - last_steering
    if abs(change) > max_change:
        return last_steering + np.sign(change) * max_change
    return new_steering

@sio.on('telemetry')
def telemetry(sid, data):
    global last_steering
    start_time = time.time()
    
    speed = float(data['speed'])
    
    # OPTIMIZED: Fast image decode and preprocess
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = image.reshape(IMAGE_SHAPE)
    
    # FAST PREDICTION
    raw_steering = float(predict_fn(image)[0][0])
    
    # SMOOTH and CLIP steering for stability
    smoothed_steering = smooth_control(raw_steering, steering_history)
    steering_angle = clip_steering_change(smoothed_steering, last_steering)
    last_steering = steering_angle
    
    # Throttle control
    raw_throttle = 1.0 - speed / speed_limit
    throttle = smooth_control(raw_throttle, throttle_history)
    
    # Performance tracking
    frame_time = (time.time() - start_time) * 1000
    frame_times.append(frame_time)
    avg_frame_time = np.mean(frame_times)
    fps = 1000 / avg_frame_time if avg_frame_time > 0 else 0
    
    print(f'Steering: {steering_angle:.4f} | Throttle: {throttle:.4f} | '
          f'Speed: {speed:.2f} | FPS: {fps:.1f} | Latency: {frame_time:.1f}ms')
    
    send_control(steering_angle, throttle)

@sio.on('connect')
def connect(sid, environ):
    global last_steering
    print('Connected to simulator')
    steering_history.clear()
    throttle_history.clear()
    frame_times.clear()
    last_steering = 0.0
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    })

if __name__ == '__main__':
    model_name = 'Dodel-m2-02'
    
    # Load model
    print("Loading model...")
    model = tf.keras.models.load_model('model/' + model_name + '.h5', compile=False)
    
    # OPTIMIZE: Create compiled prediction function
    @tf.function(reduce_retracing=True)
    def _predict(x):
        return model(x, training=False)
    
    # Warm up with dummy prediction
    print("Warming up model...")
    dummy = np.zeros(IMAGE_SHAPE, dtype=np.float32)
    _ = _predict(tf.constant(dummy))
    
    # Create fast predict function
    predict_fn = lambda x: _predict(tf.constant(x, dtype=tf.float32)).numpy()
    
    print("Model ready! Starting server...")
    print(f"Speed limit: {speed_limit}")
    print(f"Steering smoothing: {STEERING_SMOOTHING} frames")
    print(f"Max steering change: {MAX_STEERING_CHANGE}")
    
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
