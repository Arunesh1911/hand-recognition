import cv2
from sign_detector import SignLanguageDetector
from speech_converter import SpeechConverter

def main():
    detector = SignLanguageDetector()
    converter = SpeechConverter()
    cap = cv2.VideoCapture(0)
    
    print("Starting sign language detection...")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect sign language
        sign = detector.detect(frame)
        
        if sign:
            cv2.putText(frame, f"Sign: {sign}", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            converter.convert_to_speech(sign)
        
        cv2.imshow('Sign Language Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
