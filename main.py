import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import cv2
import mediapipe as mp
import numpy as np
import datetime
import time
from dotenv import load_dotenv
import threading
import sys
from queue import Queue

# Exception handling
def handle_exception(exc_type, exc_value, exc_traceback):
    print(f"Error: {exc_value}")

sys.excepthook = handle_exception

# Load environment variables
load_dotenv()

print("AI Gesture Camera initialized successfully.")

# App configuration
CONFIG = {
    "IMAGE_STYLES": ["Studio Ghibli", "Cartoon", "Sketch", "Watercolor", "Oil Painting"],
    "CAPTURE_DIR": "captures",
    "AI_GENERATED_DIR": "ai_generated",
    "TIMER_SECONDS": 3,
    "MIN_DETECTION_CONFIDENCE": 0.7,
    "MIN_TRACKING_CONFIDENCE": 0.5,
    "GESTURE_COOLDOWN": 1.0,  # Increased cooldown
    "PROCESSING_WIDTH": 640,  # Process at lower resolution for speed
    "CAMERA_WIDTH": 1280,
    "CAMERA_HEIGHT": 720,
    "CAMERA_FPS": 30,
    "SKIP_FRAMES": 2,  # Process every nth frame for better performance
}

# Create necessary directories
for path in [CONFIG["CAPTURE_DIR"], CONFIG["AI_GENERATED_DIR"]]:
    os.makedirs(path, exist_ok=True)

# Initialize MediaPipe Hand tracking with optimized settings
try:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=CONFIG["MIN_DETECTION_CONFIDENCE"],
        min_tracking_confidence=CONFIG["MIN_TRACKING_CONFIDENCE"],
        model_complexity=0  # Use simpler model for better performance
    )
    mp_drawing = mp.solutions.drawing_utils
    print("MediaPipe initialized successfully.")
except Exception as e:
    print(f"MediaPipe initialization error: {e}")
    sys.exit(1)

# App state with thread safety
class AppState:
    def __init__(self):
        self.current_mode = 0
        self.capturing = False
        self.countdown = False
        self.countdown_start = 0
        self.preview_image = None
        self.ai_generated_image = None
        self.processing_image = False
        self.last_gesture_time = 0
        self.gesture_queue = Queue()
        self.lock = threading.Lock()
        self.frame_count = 0

state = AppState()

# ================== Gesture Recognition Functions ==================

def count_fingers(hand_landmarks):
    """Count the number of fingers up"""
    tip_ids = [4, 8, 12, 16, 20]
    pip_ids = [3, 6, 10, 14, 18]
    
    fingers_up = []
    
    # Thumb (check horizontally)
    if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[pip_ids[0]].x:
        fingers_up.append(1)
    else:
        fingers_up.append(0)
    
    # Other four fingers (check vertically)
    for i in range(1, 5):
        if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[pip_ids[i]].y:
            fingers_up.append(1)
        else:
            fingers_up.append(0)
    
    return sum(fingers_up)

def detect_gesture(frame):
    """Detect gesture from frame - optimized for performance"""
    try:
        # Only process every nth frame to improve performance
        state.frame_count += 1
        if state.frame_count % CONFIG["SKIP_FRAMES"] != 0:
            return None
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Only draw landmarks if not processing to save resources
                if not state.processing_image:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2))
                
                finger_count = count_fingers(hand_landmarks)
                
                # Peace sign (index and middle fingers only)
                if (hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y and 
                    hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y and
                    hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y and
                    hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y):
                    return "peace"
                
                # Thumbs up
                elif (hand_landmarks.landmark[4].y < hand_landmarks.landmark[3].y and 
                      hand_landmarks.landmark[8].y > hand_landmarks.landmark[6].y and
                      hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y and
                      hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y and
                      hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y):
                    return "thumbs_up"
                
                # Fist (all fingers closed)
                elif finger_count == 0:
                    return "fist"
                
                # Open palm (all fingers open)
                elif finger_count == 5:
                    return "open_palm"
    except Exception as e:
        print(f"Gesture detection error: {e}")
    
    return None

# ================== Image Processing Functions (OPTIMIZED) ==================

def capture_image(frame):
    """Capture and save image"""
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(CONFIG["CAPTURE_DIR"], f"capture_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Image captured: {filename}")
        return filename
    except Exception as e:
        print(f"Image capture error: {e}")
        return None

def apply_studio_ghibli_effect(image):
    """Apply Studio Ghibli anime style effect - OPTIMIZED"""
    try:
        # Resize for faster processing
        h, w = image.shape[:2]
        small = cv2.resize(image, (CONFIG["PROCESSING_WIDTH"], 
                                   int(h * CONFIG["PROCESSING_WIDTH"] / w)))
        
        # Apply bilateral filter for smooth anime regions
        smooth = cv2.bilateralFilter(small, 5, 50, 50)
        
        # Increase saturation
        hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)
        hsv = hsv.astype(np.uint8)
        saturated = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Simple color quantization using cv2.pyrMeanShiftFiltering (faster than k-means)
        quantized = cv2.pyrMeanShiftFiltering(saturated, 15, 40)
        
        # Add edge lines
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                      cv2.THRESH_BINARY, 9, 2)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Combine
        result = cv2.bitwise_and(quantized, edges)
        result = cv2.GaussianBlur(result, (3, 3), 0)
        
        # Resize back to original size
        result = cv2.resize(result, (w, h))
        
        return result
    except Exception as e:
        print(f"Ghibli effect error: {e}")
        return image

def apply_cartoon_effect(image):
    """Apply cartoon effect - OPTIMIZED"""
    try:
        # Resize for speed
        h, w = image.shape[:2]
        small = cv2.resize(image, (CONFIG["PROCESSING_WIDTH"], 
                                   int(h * CONFIG["PROCESSING_WIDTH"] / w)))
        
        # Bilateral filter
        color = cv2.bilateralFilter(small, 5, 150, 150)
        
        # Detect edges
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                      cv2.THRESH_BINARY, 7, 2)
        
        # Combine
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cartoon = cv2.bitwise_and(color, edges)
        
        # Resize back
        cartoon = cv2.resize(cartoon, (w, h))
        
        return cartoon
    except Exception as e:
        print(f"Cartoon effect error: {e}")
        return image

def apply_sketch_effect(image):
    """Apply pencil sketch effect"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inverted = 255 - gray
        blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
        inverted_blurred = 255 - blurred
        sketch = cv2.divide(gray, inverted_blurred, scale=256.0)
        
        return sketch
    except Exception as e:
        print(f"Sketch effect error: {e}")
        return image

def apply_watercolor_effect(image):
    """Apply watercolor painting effect - OPTIMIZED"""
    try:
        # Resize for speed
        h, w = image.shape[:2]
        small = cv2.resize(image, (CONFIG["PROCESSING_WIDTH"], 
                                   int(h * CONFIG["PROCESSING_WIDTH"] / w)))
        
        # Apply bilateral filter
        result = cv2.bilateralFilter(small, 5, 75, 75)
        result = cv2.bilateralFilter(result, 5, 75, 75)
        
        # Median blur for watercolor softness
        result = cv2.medianBlur(result, 5)
        
        # Boost colors
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0, 255)
        hsv = hsv.astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Resize back
        result = cv2.resize(result, (w, h))
        
        return result
    except Exception as e:
        print(f"Watercolor effect error: {e}")
        return image

def apply_oil_painting_effect(image):
    """Apply oil painting effect - OPTIMIZED"""
    try:
        # Resize for speed
        h, w = image.shape[:2]
        small = cv2.resize(image, (CONFIG["PROCESSING_WIDTH"], 
                                   int(h * CONFIG["PROCESSING_WIDTH"] / w)))
        
        # Use stylization
        oil = cv2.stylization(small, sigma_s=60, sigma_r=0.6)
        
        # Add some detail
        detail = cv2.detailEnhance(small, sigma_s=10, sigma_r=0.15)
        result = cv2.addWeighted(oil, 0.7, detail, 0.3, 0)
        
        # Resize back
        result = cv2.resize(result, (w, h))
        
        return result
    except Exception as e:
        print(f"Oil painting effect error: {e}")
        return image

def generate_ai_image(image_path, style):
    """Generate AI styled image using OpenCV filters"""
    try:
        print(f"Generating {style} style...")
        
        # Load original image
        original_image = cv2.imread(image_path)
        
        if original_image is None:
            print(f"Failed to load image: {image_path}")
            return None
        
        # Apply appropriate effect based on style
        effect_map = {
            "Studio Ghibli": apply_studio_ghibli_effect,
            "Cartoon": apply_cartoon_effect,
            "Sketch": apply_sketch_effect,
            "Watercolor": apply_watercolor_effect,
            "Oil Painting": apply_oil_painting_effect
        }
        
        effect_func = effect_map.get(style)
        if effect_func:
            result_image = effect_func(original_image)
        else:
            print(f"Unknown style: {style}")
            return None
        
        # Save the result
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(CONFIG["AI_GENERATED_DIR"], 
                                   f"ai_{style.replace(' ', '_')}_{timestamp}.jpg")
        cv2.imwrite(output_path, result_image)
        
        print(f"✓ AI image saved: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"AI image generation error: {e}")
        import traceback
        traceback.print_exc()
        return None

# ================== UI Drawing Functions ==================

def draw_ui(frame, gesture):
    """Draw UI elements - optimized for performance"""
    try:
        h, w, _ = frame.shape
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Show current mode
        mode_text = f"Mode: {CONFIG['IMAGE_STYLES'][state.current_mode]}"
        cv2.putText(frame, mode_text, (10, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        
        # Show detected gesture
        if gesture:
            cv2.putText(frame, f"Gesture: {gesture}", (10, 75), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Show instructions
        cv2.putText(frame, "Peace: Capture | Fist: Change Mode | Thumbs Up: Timer", 
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show processing indicator
        if state.processing_image:
            # Animated processing indicator
            dots = "." * (int(time.time() * 2) % 4)
            cv2.putText(frame, f"Processing{dots}", (w - 250, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        
        # Draw countdown
        if state.countdown:
            remaining = CONFIG["TIMER_SECONDS"] - (time.time() - state.countdown_start)
            if remaining > 0:
                # Draw countdown circle
                center = (w // 2, h // 2)
                radius = 100
                cv2.circle(frame, center, radius, (0, 0, 255), 10)
                
                # Draw countdown number
                countdown_num = int(remaining + 1)
                cv2.putText(frame, str(countdown_num), 
                           (center[0] - 40, center[1] + 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 8)
                return False
            else:
                return True  # Signal to capture
        
        return False
    except Exception as e:
        print(f"UI drawing error: {e}")
        return False

def display_images(frame, preview, ai_image):
    """Display original and AI generated images side by side"""
    try:
        if preview is None or ai_image is None:
            return frame
        
        h, w, _ = frame.shape
        
        # Resize images to fit
        preview_resized = cv2.resize(preview, (w // 2, h))
        ai_resized = cv2.resize(ai_image, (w // 2, h))
        
        # Combine images horizontally
        combined = np.hstack((preview_resized, ai_resized))
        
        # Add labels with background
        cv2.rectangle(combined, (5, 5), (200, 50), (0, 0, 0), -1)
        cv2.rectangle(combined, (w // 2 + 5, 5), (w // 2 + 350, 50), (0, 0, 0), -1)
        
        cv2.putText(combined, "Original", (10, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, f"AI: {CONFIG['IMAGE_STYLES'][state.current_mode]}", 
                    (w // 2 + 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Add instruction to go back
        cv2.putText(combined, "Press SPACE to continue", 
                    (w // 2 - 150, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return combined
    except Exception as e:
        print(f"Image display error: {e}")
        return frame

# ================== Background Processing ==================

def process_image_in_background(image_path, style):
    """Generate AI image in background thread"""
    try:
        with state.lock:
            state.processing_image = True
        
        print(f"Starting {style} processing...")
        ai_image_path = generate_ai_image(image_path, style)
        
        if ai_image_path:
            ai_img = cv2.imread(ai_image_path)
            if ai_img is not None:
                with state.lock:
                    state.ai_generated_image = ai_img
                print("✓ Processing complete!")
            else:
                print("Failed to load generated image.")
        else:
            print("Failed to generate AI image.")
        
    except Exception as e:
        print(f"Background processing error: {e}")
    finally:
        with state.lock:
            state.processing_image = False

# ================== Main App ==================

def main():
    try:
        # Start webcam with optimized settings
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        
        # Set camera resolution and FPS
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["CAMERA_WIDTH"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["CAMERA_HEIGHT"])
        cap.set(cv2.CAP_PROP_FPS, CONFIG["CAMERA_FPS"])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size to minimize lag
        
        print("\n" + "="*50)
        print("🎨 AI GESTURE CAMERA")
        print("="*50)
        print("\nGesture Controls:")
        print("  ✌️  Peace Sign  → Capture photo")
        print("  ✊  Fist       → Change style mode")
        print("  👍  Thumbs Up  → Start 3-second timer")
        print("\nKeyboard Controls:")
        print("  'q'     → Quit")
        print("  SPACE   → Return to camera")
        print("="*50 + "\n")
        
        processing_thread = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect gesture
            gesture = detect_gesture(frame)
            
            # Handle gestures with cooldown
            current_time = time.time()
            can_process_gesture = (current_time - state.last_gesture_time) > CONFIG["GESTURE_COOLDOWN"]
            
            if gesture and not state.processing_image and can_process_gesture:
                
                if gesture == "peace" and not state.capturing and not state.countdown:
                    state.last_gesture_time = current_time
                    state.capturing = True
                    
                    image_path = capture_image(frame)
                    
                    if image_path:
                        with state.lock:
                            state.preview_image = cv2.imread(image_path)
                        
                        # Start processing in background
                        processing_thread = threading.Thread(
                            target=process_image_in_background,
                            args=(image_path, CONFIG['IMAGE_STYLES'][state.current_mode]),
                            daemon=True
                        )
                        processing_thread.start()
                    
                    state.capturing = False
                
                elif gesture == "fist" and not state.capturing and not state.countdown:
                    state.last_gesture_time = current_time
                    state.current_mode = (state.current_mode + 1) % len(CONFIG["IMAGE_STYLES"])
                    print(f"\n>>> Mode: {CONFIG['IMAGE_STYLES'][state.current_mode]} <<<\n")
                
                elif gesture == "thumbs_up" and not state.countdown and not state.capturing:
                    state.last_gesture_time = current_time
                    state.countdown = True
                    state.countdown_start = time.time()
                    print("⏱️  Timer started...")
            
            # Draw UI and check for capture
            capture_now = draw_ui(frame, gesture)
            
            # Capture when countdown finishes
            if capture_now and not state.processing_image:
                state.countdown = False
                image_path = capture_image(frame)
                
                if image_path:
                    with state.lock:
                        state.preview_image = cv2.imread(image_path)
                    
                    # Start processing in background
                    processing_thread = threading.Thread(
                        target=process_image_in_background,
                        args=(image_path, CONFIG['IMAGE_STYLES'][state.current_mode]),
                        daemon=True
                    )
                    processing_thread.start()
            
            # Display images
            with state.lock:
                if state.preview_image is not None and state.ai_generated_image is not None:
                    display_frame = display_images(frame, state.preview_image, state.ai_generated_image)
                else:
                    display_frame = frame
            
            # Show frame
            cv2.imshow('AI Gesture Camera', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space bar to return to camera view
                with state.lock:
                    state.preview_image = None
                    state.ai_generated_image = None
                print("Returned to camera view")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Application closed\n")
        
    except Exception as e:
        print(f"Main application error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()