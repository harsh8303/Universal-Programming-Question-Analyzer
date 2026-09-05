import easyocr
import cv2

# Initialize reader (gpu=False is safer on Mac CPU, we rely on resizing for speed)
reader = easyocr.Reader(['en'], gpu=False)

def extract_text(image_path):
    # 1. Read the image using OpenCV (installed automatically with EasyOCR)
    img = cv2.imread(image_path)
    
    if img is None:
        return ""

    # 2. DOWN-SCALE THE IMAGE (The Magic Speed Fix!)
    # Mac retina screenshots are massive. Shrinking them to a max width of 1000px 
    # makes EasyOCR run 10x-20x faster without losing text readability.
    max_width = 1000
    height, width = img.shape[:2]
    
    if width > max_width:
        scaling_factor = max_width / float(width)
        img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    # Optional: Convert to grayscale to make text pop out more and process faster
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Extract text from the optimized image
    results = reader.readtext(gray, detail=0)
    
    # Join all detected text pieces into a single string
    extracted_text = " ".join(results)
    
    return extracted_text