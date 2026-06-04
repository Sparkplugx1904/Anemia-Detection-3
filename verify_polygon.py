import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

# --- CONFIGURATION (MATCHING ANDROID SEGMENTOR.KT) ---
MODEL_PATH = "AnedetApp/app/src/main/assets/yolo26n_seg_fp16.tflite"
IMAGE_PATH = "test_eye.jpg"
INPUT_SIZE = 320
CONF_THRESHOLD = 0.25

def letterbox(img, new_shape=(320, 320), color=(128, 128, 128)):
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def run_verification():
    # 1. Load Model
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 2. Preprocess
    img_orig = cv2.imread(IMAGE_PATH)
    img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    img_pad, scale, (pad_w, pad_h) = letterbox(img_rgb, (INPUT_SIZE, INPUT_SIZE))
    
    input_data = np.expand_dims(img_pad.astype(np.float32) / 255.0, axis=0)
    
    # 3. Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Output 0: Detection [1, 300, 38]
    det = interpreter.get_tensor(output_details[0]['index'])
    # Output 1: Proto [1, 80, 80, 32]
    proto = interpreter.get_tensor(output_details[1]['index'])

    print(f"Model Detection Shape: {det.shape}")
    print(f"Model Proto Shape: {proto.shape}")

    # Format detection
    det = det[0]
    num_anchors = det.shape[0]
    num_features = det.shape[1]
    
    # 4. Parse Detection (x1y1x2y2 format for nc=1)
    # Based on 38 features: likely [x1, y1, x2, y2, conf, class0, mask0...mask31]
    conf_scores = det[:, 4]
    best_idx = np.argmax(conf_scores)
    best_conf = conf_scores[best_idx]
    
    print(f"Best Detection Confidence: {best_conf:.2f}")

    x1, y1, x2, y2 = det[best_idx, 0:4]
    # Coefficients likely start at index 6 if index 4 is conf and 5 is class
    coeffs = det[best_idx, 6:38] 

    # Auto-scale normalized coords
    if x1 <= 1.01 and x2 <= 1.01:
        x1 *= INPUT_SIZE; x2 *= INPUT_SIZE; y1 *= INPUT_SIZE; y2 *= INPUT_SIZE

    # Decode to Original Space
    orig_h, orig_w = img_orig.shape[:2]
    x1_orig = (x1 - pad_w) / scale
    y1_orig = (y1 - pad_h) / scale
    x2_orig = (x2 - pad_w) / scale
    y2_orig = (y2 - pad_h) / scale

    # 5. Compute Mask
    proto_h, proto_w = proto.shape[1], proto.shape[2]
    proto_data = proto[0] # [80, 80, 32]
    mask_proto = np.dot(proto_data, coeffs)
    
    mask_sigmoid = sigmoid(mask_proto)
    binary_mask = (mask_sigmoid > 0.5).astype(np.uint8)

    # Spatial Crop to Bbox (Map 320 space to 80 space)
    p_left = int(x1 * (proto_w/320))
    p_top = int(y1 * (proto_h/320))
    p_right = int(x2 * (proto_w/320))
    p_bottom = int(y2 * (proto_h/320))
    
    crop_mask = np.zeros_like(binary_mask)
    crop_mask[p_top:p_bottom, p_left:p_right] = binary_mask[p_top:p_bottom, p_left:p_right]

    # 6. Extract Polygon
    contours, _ = cv2.findContours(crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("FAILED: No polygon contours found")
        return

    main_contour = max(contours, key=cv2.contourArea)
    
    # Map back to original space
    polygon_points = []
    for pt in main_contour:
        px, py = pt[0]
        mx = px * (320.0 / proto_w)
        my = py * (320.0 / proto_h)
        ox = (mx - pad_w) / scale
        oy = (my - pad_h) / scale
        polygon_points.append([float(ox), float(oy)])

    print("\n--- POLYGON COORDINATES (ORIGINAL IMAGE SPACE) ---")
    print(np.array(polygon_points))
    print(f"Total Points: {len(polygon_points)}")

    # 7. Visualization
    plt.figure(figsize=(12, 6))
    
    # Subplot 1: Raw Image + Mask Overlay
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    
    # Draw Mask (Scaled up)
    mask_full = cv2.resize(crop_mask, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_NEAREST)
    mask_colored = np.zeros_like(img_pad)
    mask_colored[mask_full > 0] = [76, 175, 80] # Android Green
    
    # Re-crop padding for display
    vis_img = img_pad.copy()
    cv2.addWeighted(mask_colored, 0.5, vis_img, 0.5, 0, vis_img)
    plt.imshow(vis_img[int(pad_h):int(INPUT_SIZE-pad_h), int(pad_w):int(INPUT_SIZE-pad_w)])
    plt.title("Android Logic Simulation (Live Preview)")
    plt.axis('off')

    # Subplot 2: Precision Polygon Trace
    plt.subplot(1, 2, 2)
    plt.imshow(img_rgb)
    poly_arr = np.array(polygon_points)
    plt.plot(poly_arr[:, 0], poly_arr[:, 1], color='lime', linewidth=2, label='Polygon Trace')
    plt.scatter(poly_arr[:, 0], poly_arr[:, 1], color='red', s=5)
    plt.gca().add_patch(plt.Rectangle((x1_orig, y1_orig), x2_orig-x1_orig, y2_orig-y1_orig, 
                                      fill=False, color='yellow', linestyle='--', label='Bbox'))
    plt.title("Extracted Polygon Trace (Original Space)")
    plt.legend()
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_verification()
