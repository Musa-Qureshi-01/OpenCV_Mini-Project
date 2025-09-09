# selective_stylizer.py
import cv2
import numpy as np
import argparse
import os

def apply_clahe_bgr(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def create_subject_mask(img, debug=False):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.sum(th == 255) < np.sum(th == 0):
        th = cv2.bitwise_not(th)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((h, w), dtype=np.uint8)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, -1)
    else:
        mask = opened
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask = cv2.dilate(mask, kernel2, iterations=1)
    mask_float = mask.astype(np.float32) / 255.0
    mask_blur = cv2.GaussianBlur(mask_float, (51, 51), 20)
    if debug:
        return mask, mask_blur
    return mask_blur

def stylize(img, alpha_mask, clahe_img, params):
    h, w = img.shape[:2]
    mask_uint8 = (alpha_mask * 255).astype(np.uint8)
    
    bilateral = cv2.bilateralFilter(img, d=params['bilateral_d'], sigmaColor=params['bilateral_sc'], sigmaSpace=params['bilateral_ss'])
    subject_smooth = cv2.bitwise_and(bilateral, bilateral, mask=mask_uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    s_ch = (s_ch * params['bg_desat']).astype(np.uint8)
    hsv_desat = cv2.merge((h_ch, s_ch, v_ch))
    bg_desat = cv2.cvtColor(hsv_desat, cv2.COLOR_HSV2BGR)
    bg_blur = cv2.GaussianBlur(bg_desat, (params['bg_blur_k'], params['bg_blur_k']), params['bg_blur_sigma'])
    alpha_3c = np.dstack([alpha_mask]*3)
    composed = (subject_smooth.astype(np.float32) * alpha_3c + bg_blur.astype(np.float32) * (1.0 - alpha_3c)).astype(np.uint8)
    gray_enh = cv2.cvtColor(clahe_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_enh, params['canny_min'], params['canny_max'])
    if params['use_hough']:
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=30, maxLineGap=10)
        if lines is not None:
            for x1, y1, x2, y2 in lines[:,0]:
                cv2.line(edges, (x1,y1), (x2,y2), 255, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    edges_dil = cv2.dilate(edges, kernel, iterations=1)
    edges_subject = cv2.bitwise_and(edges_dil, mask_uint8)
    edges_col = cv2.cvtColor(edges_subject, cv2.COLOR_GRAY2BGR)
    edges_col = (edges_col // 255) * np.array(params['edge_color'], dtype=np.uint8)
    edge_mask_f = (edges_subject.astype(np.float32) / 255.0)[:,:,None]
    final = (composed.astype(np.float32) * (1.0 - edge_mask_f) + edges_col.astype(np.float32) * edge_mask_f).astype(np.uint8)
    final_hsv = cv2.cvtColor(final, cv2.COLOR_BGR2HSV).astype(np.float32)
    h2, s2, v2 = cv2.split(final_hsv)
    s2 = np.clip(s2 * (1.0 + params['subject_vibrance']), 0, 255)
    final_hsv = cv2.merge((h2, s2, v2)).astype(np.uint8)
    final_vibrant = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    final_out = (subject_smooth.astype(np.float32) * alpha_3c + final_vibrant.astype(np.float32) * (1.0 - alpha_3c)).astype(np.uint8)
    return final_out, mask_uint8, edges_subject

def make_montage(original, mask_uint8, enhanced, final, max_width=1600):
    mask_color = cv2.addWeighted(original, 0.6, np.dstack([mask_uint8, np.zeros_like(mask_uint8), np.zeros_like(mask_uint8)]), 0.4, 0)
    h, w = original.shape[:2]
    imgs = [original, mask_color, enhanced, final]
    imgs_resized = [cv2.resize(im, (w, h)) for im in imgs]
    row = np.hstack(imgs_resized)
    if row.shape[1] > max_width:
        scale = max_width / row.shape[1]
        row = cv2.resize(row, (int(row.shape[1] * scale), int(row.shape[0] * scale)))
    return row

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=False, default="image.jpg", help="input image path")
    ap.add_argument("--output", "-o", required=False, default="stylized_result.jpg", help="output path")
    return vars(ap.parse_args())

def main():
    args = parse_args()
    img_path = r'C:\Users\musaq\OneDrive\Desktop\Intership - G35 Python\Intership - Training\OpenCV_MiniProject_09-09\image.png'
    out_path = args['output']
    if not os.path.exists(img_path):
        print(f"[ERROR] Input not found: {img_path}")
        return
    img = cv2.imread(img_path)
    if img is None:
        print("[ERROR] Could not read image.")
        return
    max_dim = 1200
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    enhanced = apply_clahe_bgr(img)
    alpha_mask = create_subject_mask(enhanced)
    params = {
        'bilateral_d': 11,
        'bilateral_sc': 75,
        'bilateral_ss': 75,
        'bg_desat': 0.15,
        'bg_blur_k': 31,
        'bg_blur_sigma': 25,
        'canny_min': 80,
        'canny_max': 180,
        'use_hough': False,
        'edge_color': (10, 10, 10),
        'subject_vibrance': 0.18,
    }
    final, mask_uint8, edges_subject = stylize(img, alpha_mask, enhanced, params)
    cv2.imwrite(out_path, final)
    print(f"[OK] Saved stylized output to: {out_path}")
    montage = make_montage(img, mask_uint8, enhanced, final)
    cv2.imshow("Original | Mask preview | Enhanced | Final", montage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
