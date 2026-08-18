import cv2
import numpy as np
import urllib.request
import os

base_dir = '/home/chakradhar/Downloads/NEW_PROJECT/OpenCV_Experiments'

# Download sample building image
img_path = 'building.jpg'
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)
urllib.request.urlretrieve('https://raw.githubusercontent.com/opencv/opencv/master/samples/data/building.jpg', img_path)
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

experiments = [
    "Exp_01_Grayscale", "Exp_02_Blur", "Exp_03_Canny", "Exp_04_Dilate", "Exp_05_Erode",
    "Exp_06_VideoProcessing", "Exp_07_WebcamCapture", "Exp_08_Scaling", "Exp_09_Rotation",
    "Exp_10_Translation", "Exp_11_Affine", "Exp_12_Perspective", "Exp_13_PerspectiveVideo",
    "Exp_14_Homography", "Exp_15_DLT", "Exp_16_CannyMethod", "Exp_17_SobelX", "Exp_18_SobelY",
    "Exp_19_SobelXY", "Exp_20_LaplacianNegativeCenter", "Exp_21_LaplacianExtendedDiagonals",
    "Exp_22_LaplacianPositiveCenter", "Exp_23_UnsharpMasking", "Exp_24_HighBoost",
    "Exp_25_GradientMasking", "Exp_26_Watermarking", "Exp_27_CropCopyPaste"
]

def get_out(idx):
    return os.path.join(base_dir, experiments[idx-1], 'output.jpg')

# 1
cv2.imwrite(get_out(1), gray)

# 2
blurred = cv2.GaussianBlur(img, (15, 15), 0)
cv2.imwrite(get_out(2), blurred)

# 3
edges = cv2.Canny(img, 100, 200)
cv2.imwrite(get_out(3), edges)

# 4
kernel = np.ones((5,5), np.uint8)
dilated = cv2.dilate(edges, kernel, iterations=1)
cv2.imwrite(get_out(4), dilated)

# 5
eroded = cv2.erode(dilated, kernel, iterations=1)
cv2.imwrite(get_out(5), eroded)

# 6
height, width = 480, 640
out = cv2.VideoWriter('dummy_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (width, height))
for i in range(10):
    frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    out.write(frame)
out.release()
cap = cv2.VideoCapture('dummy_video.avi')
ret, frame = cap.read()
if ret:
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(get_out(6), gray_frame)
cap.release()

# 7 WebCam - Using user uploaded photo
user_photo = cv2.imread('/home/chakradhar/.gemini/antigravity-ide/brain/0e089800-7e5d-4542-9630-b9fe125f9c75/media__1787062663404.jpg')
if user_photo is not None:
    cv2.imwrite(get_out(7), user_photo)
else:
    dummy_webcam = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(dummy_webcam, 'Webcam Not Available', (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imwrite(get_out(7), dummy_webcam)

# 8
smaller = cv2.resize(img, (0,0), fx=0.1, fy=0.1)
bigger = cv2.resize(img, (0,0), fx=3.0, fy=3.0)
cv2.imwrite(get_out(8), smaller) # Just save one for the preview

# 9
(h, w) = img.shape[:2]
center = (w // 2, h // 2)
M_cw = cv2.getRotationMatrix2D(center, -45, 1.0)
rotated_cw = cv2.warpAffine(img, M_cw, (w, h))
cv2.imwrite(get_out(9), rotated_cw)

# 10
M_translate = np.float32([[1, 0, 150], [0, 1, 250]])
translated = cv2.warpAffine(img, M_translate, (w, h))
cv2.imwrite(get_out(10), translated)

# 11
pts1 = np.float32([[50,50], [200,50], [50,200]])
pts2 = np.float32([[10,100], [200,50], [100,250]])
M_affine = cv2.getAffineTransform(pts1, pts2)
affine_img = cv2.warpAffine(img, M_affine, (w, h))
cv2.imwrite(get_out(11), affine_img)

# 12
pts1_p = np.float32([[56,65], [368,52], [28,387], [389,390]])
pts2_p = np.float32([[0,0], [300,0], [0,300], [300,300]])
M_persp = cv2.getPerspectiveTransform(pts1_p, pts2_p)
perspective_img = cv2.warpPerspective(img, M_persp, (300,300))
cv2.imwrite(get_out(12), perspective_img)

# 13
cap = cv2.VideoCapture('dummy_video.avi')
ret, frame = cap.read()
if ret:
    h_v, w_v = frame.shape[:2]
    pts1_v = np.float32([[0,0], [w_v,0], [0,h_v], [w_v,h_v]])
    pts2_v = np.float32([[50,50], [w_v-50,0], [0,h_v-50], [w_v-50,h_v-50]])
    M_v = cv2.getPerspectiveTransform(pts1_v, pts2_v)
    dst_v = cv2.warpPerspective(frame, M_v, (w_v, h_v))
    cv2.imwrite(get_out(13), dst_v)

# 14
src_pts = np.float32([[0,0], [w,0], [0,h], [w,h]])
dst_pts = np.float32([[150,150], [w-150, 0], [0, h-150], [w-150, h-150]])
H, status = cv2.findHomography(src_pts, dst_pts)
homography_img = cv2.warpPerspective(img, H, (w, h))
cv2.imwrite(get_out(14), homography_img)

# 15
def dlt_homography(src, dst):
    A = []
    for i in range(4):
        x, y = src[i][0], src[i][1]
        u, v = dst[i][0], dst[i][1]
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    A = np.asarray(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1,:] / Vh[-1,-1]
    return L.reshape(3, 3)
H_dlt = dlt_homography(src_pts, dst_pts)
dlt_img = cv2.warpPerspective(img, H_dlt, (w,h))
cv2.imwrite(get_out(15), dlt_img)

# 16
edges_canny = cv2.Canny(gray, 100, 200)
cv2.imwrite(get_out(16), edges_canny)

# 17
sobelx = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5))
cv2.imwrite(get_out(17), sobelx)

# 18
sobely = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5))
cv2.imwrite(get_out(18), sobely)

# 19
sobelxy = cv2.bitwise_or(sobelx, sobely)
cv2.imwrite(get_out(19), sobelxy)

# 20
kernel_neg = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
lap_neg = cv2.filter2D(gray, cv2.CV_64F, kernel_neg)
sharp_neg = np.clip(gray - lap_neg, 0, 255).astype(np.uint8)
cv2.imwrite(get_out(20), sharp_neg)

# 21
kernel_diag = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
lap_diag = cv2.filter2D(gray, cv2.CV_64F, kernel_diag)
sharp_diag = np.clip(gray - lap_diag, 0, 255).astype(np.uint8)
cv2.imwrite(get_out(21), sharp_diag)

# 22
kernel_pos = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharp_pos = cv2.filter2D(gray, -1, kernel_pos)
cv2.imwrite(get_out(22), sharp_pos)

# 23
gaussian_blur = cv2.GaussianBlur(img, (15,15), 20.0)
unsharp_img = cv2.addWeighted(img, 2.5, gaussian_blur, -1.5, 0)
cv2.imwrite(get_out(23), unsharp_img)

# 24
high_boost_img = cv2.addWeighted(img, 1.2, gaussian_blur, -1, 0)
cv2.imwrite(get_out(24), high_boost_img)

# 25
sobel_mag = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
sobel_mag_bgr = cv2.cvtColor(sobel_mag, cv2.COLOR_GRAY2BGR)
gradient_sharpened = cv2.addWeighted(img, 1.0, sobel_mag_bgr, 0.5, 0)
cv2.imwrite(get_out(25), gradient_sharpened)

# 26
watermark = np.zeros_like(img)
cv2.putText(watermark, 'COLAB WATERMARK', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
watermarked = cv2.addWeighted(img, 1.0, watermark, 0.3, 0)
cv2.imwrite(get_out(26), watermarked)

# 27
pasted = img.copy()
cropped = img[100:300, 100:300]
if cropped.size > 0:
    pasted[0:200, 0:200] = cropped
cv2.imwrite(get_out(27), pasted)
print('Done!')
