import numpy as np
import cv2

def enhance_image(img):
    """Enhance contrast using CLAHE."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def get_subject_mask(img):
    """Generate binary mask using largest contour."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert if background is brighter
    if np.sum(thresh == 255) > np.sum(thresh == 0):
        thresh = cv2.bitwise_not(thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(gray)

    c = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [c], -1, 255, -1)

    # Gentle smoothing
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    mask = cv2.GaussianBlur(mask, (5, 5), 2)
    return mask

def stylize_image(img, mask):
    """Stylize subject vs. background."""
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Subject (smoothed)
    subject = cv2.bitwise_and(img, img, mask=mask)
    subject_smooth = cv2.bilateralFilter(subject, d=9, sigmaColor=75, sigmaSpace=75)

    # Background (slightly desaturated + blurred)
    background = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
    hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = (s * 0.4).astype(np.uint8)  # keep some color
    background_desat = cv2.merge((h, s, v))
    background_desat = cv2.cvtColor(background_desat, cv2.COLOR_HSV2BGR)
    background_blur = cv2.GaussianBlur(background_desat, (21, 21), 10)

    # Blend subject + background
    combined = cv2.addWeighted(subject_smooth, 1, background_blur, 1, 0)

    # Edges (light sketch effect only on subject)
    edges = cv2.Canny(img, 100, 200)
    edges = cv2.dilate(edges, np.ones((2,2), np.uint8))
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_masked = cv2.bitwise_and(edges_colored, mask_3ch)
    final = cv2.addWeighted(combined, 1.0, edges_masked, 0.15, 0)

    return final, edges

def make_montage(original, mask, enhanced, final):
    """Create 2x2 montage."""
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    h, w = original.shape[:2]
    enhanced_resized = cv2.resize(enhanced, (w, h))
    final_resized = cv2.resize(final, (w, h))
    top = np.hstack((original, mask_bgr))
    bottom = np.hstack((enhanced_resized, final_resized))
    return np.vstack((top, bottom))

def main():
    img = cv2.imread(r"C:\Users\musaq\OneDrive\Desktop\Intership - G35 Python\Intership - Training\OpenCV_MiniProject_09-09\image.png")
    if img is None:
        print("Error loading image")
        return

    enhanced = enhance_image(img)
    mask = get_subject_mask(enhanced)
    final, edges = stylize_image(enhanced, mask)

    montage = make_montage(img, mask, enhanced, final)

    cv2.imshow("Selective Stylizer Result", montage)
    cv2.imwrite("Stylized_result.png", final)
    cv2.imwrite("Stylized_montage.png", montage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
