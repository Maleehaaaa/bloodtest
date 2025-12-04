import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Function to classify RBC based on shape/roundness
def classify_rbc(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        roundness = 4 * np.pi * (area / (perimeter**2))
        if roundness > 0.8:
            results.append("Normal RBC")
        else:
            results.append("Thalassemia minor RBC")

    if len(results) == 0:
        return "No cells detected"

    if results.count("Normal RBC") >= results.count("Thalassemia minor RBC"):
        return "Predicted: Normal RBC"
    else:
        return "Predicted: Thalassemia minor RBC"

# ----------------------------
# Open file dialog for user to select any image
# ----------------------------
Tk().withdraw()  # Hide the main Tk window
file_path = askopenfilename(title="Select an RBC image", filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")])

if not file_path:
    print("No file selected!")
else:
    img = cv2.imread(file_path)
    if img is None:
        print("Could not load image!")
    else:
        prediction = classify_rbc(img)

        # Overlay prediction text on the image
        img_display = img.copy()
        cv2.putText(img_display, prediction, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("RBC Sample", img_display)
        print(file_path.split("/")[-1], "->", prediction)
        cv2.waitKey(0)  # Wait until key press
        cv2.destroyAllWindows()
