import os
import json

base_dir = "/home/chakradhar/Downloads/NEW_PROJECT/OpenCV_Experiments"
os.makedirs(base_dir, exist_ok=True)

experiments = [
    ("Exp_01_Grayscale", "gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\nprint('1. Grayscale Image:')\ncv2_imshow(gray_img)"),
    ("Exp_02_Blur", "blurred_img = cv2.GaussianBlur(img, (15, 15), 0)\nprint('2. Blurred Image:')\ncv2_imshow(blurred_img)"),
    ("Exp_03_Canny", "edges = cv2.Canny(img, 100, 200)\nprint('3. Outline using Canny:')\ncv2_imshow(edges)"),
    ("Exp_04_Dilate", "edges = cv2.Canny(img, 100, 200)\nkernel = np.ones((5,5), np.uint8)\ndilated_img = cv2.dilate(edges, kernel, iterations=1)\nprint('4. Dilated Image:')\ncv2_imshow(dilated_img)"),
    ("Exp_05_Erode", "edges = cv2.Canny(img, 100, 200)\nkernel = np.ones((5,5), np.uint8)\ndilated_img = cv2.dilate(edges, kernel, iterations=1)\neroded_img = cv2.erode(dilated_img, kernel, iterations=1)\nprint('5. Eroded Image:')\ncv2_imshow(eroded_img)"),
    ("Exp_06_VideoProcessing", "height, width = 480, 640\nout = cv2.VideoWriter('dummy_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (width, height))\nfor i in range(50):\n    frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)\n    out.write(frame)\nout.release()\n\ncap = cv2.VideoCapture('dummy_video.avi')\nframes_processed = 0\nwhile cap.isOpened() and frames_processed < 5:\n    ret, frame = cap.read()\n    if not ret: break\n    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)\n    print(f'Processed Frame {frames_processed+1}:')\n    cv2_imshow(gray_frame)\n    frames_processed += 1\ncap.release()"),
    ("Exp_07_WebcamCapture", "from IPython.display import display, Javascript\nfrom google.colab.output import eval_js\nfrom base64 import b64decode\n\ndef take_photo(filename='photo.jpg', quality=0.8):\n  js = Javascript('''\n    async function takePhoto(quality) {\n      const div = document.createElement('div');\n      const capture = document.createElement('button');\n      capture.textContent = 'Capture';\n      div.appendChild(capture);\n      const video = document.createElement('video');\n      video.style.display = 'block';\n      const stream = await navigator.mediaDevices.getUserMedia({video: true});\n      document.body.appendChild(div);\n      div.appendChild(video);\n      video.srcObject = stream;\n      await video.play();\n      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);\n      await new Promise((resolve) => capture.onclick = resolve);\n      const canvas = document.createElement('canvas');\n      canvas.width = video.videoWidth;\n      canvas.height = video.videoHeight;\n      canvas.getContext('2d').drawImage(video, 0, 0);\n      stream.getVideoTracks()[0].stop();\n      div.remove();\n      return canvas.toDataURL('image/jpeg', quality);\n    }\n    ''')\n  display(js)\n  data = eval_js('takePhoto({})'.format(quality))\n  binary = b64decode(data.split(',')[1])\n  with open(filename, 'wb') as f:\n    f.write(binary)\n  return filename\n\ntry:\n  filename = take_photo()\n  print('Saved to {}'.format(filename))\n  display(cv2_imshow(cv2.imread(filename)))\nexcept Exception as err:\n  print(str(err))"),
    ("Exp_08_Scaling", "smaller = cv2.resize(img, (0,0), fx=0.5, fy=0.5)\nprint('8. Smaller Image:')\ncv2_imshow(smaller)\nbigger = cv2.resize(img, (0,0), fx=2.0, fy=2.0)\nprint('8. Bigger Image:')\ncv2_imshow(bigger)"),
    ("Exp_09_Rotation", "(h, w) = img.shape[:2]\ncenter = (w // 2, h // 2)\nM_cw = cv2.getRotationMatrix2D(center, -45, 1.0)\nrotated_cw = cv2.warpAffine(img, M_cw, (w, h))\nprint('9. Clockwise Rotation (-45 deg):')\ncv2_imshow(rotated_cw)\nM_ccw = cv2.getRotationMatrix2D(center, 45, 1.0)\nrotated_ccw = cv2.warpAffine(img, M_ccw, (w, h))\nprint('9. Counter-Clockwise Rotation (45 deg):')\ncv2_imshow(rotated_ccw)"),
    ("Exp_10_Translation", "(h, w) = img.shape[:2]\nM_translate = np.float32([[1, 0, 50], [0, 1, 100]])\ntranslated = cv2.warpAffine(img, M_translate, (w, h))\nprint('10. Translated Image:')\ncv2_imshow(translated)"),
    ("Exp_11_Affine", "(h, w) = img.shape[:2]\npts1 = np.float32([[50,50], [200,50], [50,200]])\npts2 = np.float32([[10,100], [200,50], [100,250]])\nM_affine = cv2.getAffineTransform(pts1, pts2)\naffine_img = cv2.warpAffine(img, M_affine, (w, h))\nprint('11. Affine Transformation:')\ncv2_imshow(affine_img)"),
    ("Exp_12_Perspective", "pts1_p = np.float32([[56,65], [368,52], [28,387], [389,390]])\npts2_p = np.float32([[0,0], [300,0], [0,300], [300,300]])\nM_persp = cv2.getPerspectiveTransform(pts1_p, pts2_p)\nperspective_img = cv2.warpPerspective(img, M_persp, (300,300))\nprint('12. Perspective Transformation:')\ncv2_imshow(perspective_img)"),
    ("Exp_13_PerspectiveVideo", "height, width = 480, 640\nout = cv2.VideoWriter('dummy_video2.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (width, height))\nfor i in range(10):\n    frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)\n    out.write(frame)\nout.release()\n\ncap = cv2.VideoCapture('dummy_video2.avi')\nframes_processed = 0\nwhile cap.isOpened() and frames_processed < 3:\n    ret, frame = cap.read()\n    if not ret: break\n    h_v, w_v = frame.shape[:2]\n    pts1_v = np.float32([[0,0], [w_v,0], [0,h_v], [w_v,h_v]])\n    pts2_v = np.float32([[50,50], [w_v-50,0], [0,h_v-50], [w_v-50,h_v-50]])\n    M_v = cv2.getPerspectiveTransform(pts1_v, pts2_v)\n    dst_v = cv2.warpPerspective(frame, M_v, (w_v, h_v))\n    print(f'13. Perspective Video Frame {frames_processed+1}:')\n    cv2_imshow(dst_v)\n    frames_processed += 1\ncap.release()"),
    ("Exp_14_Homography", "(h, w) = img.shape[:2]\nsrc_pts = np.float32([[0,0], [w,0], [0,h], [w,h]])\ndst_pts = np.float32([[50,50], [w-50, 0], [0, h-50], [w-50, h-50]])\nH, status = cv2.findHomography(src_pts, dst_pts)\nhomography_img = cv2.warpPerspective(img, H, (w, h))\nprint('14. Homography Transformation:')\ncv2_imshow(homography_img)"),
    ("Exp_15_DLT", "(h, w) = img.shape[:2]\nsrc_pts = np.float32([[0,0], [w,0], [0,h], [w,h]])\ndst_pts = np.float32([[50,50], [w-50, 0], [0, h-50], [w-50, h-50]])\ndef dlt_homography(src, dst):\n    A = []\n    for i in range(4):\n        x, y = src[i][0], src[i][1]\n        u, v = dst[i][0], dst[i][1]\n        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])\n        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])\n    A = np.asarray(A)\n    U, S, Vh = np.linalg.svd(A)\n    L = Vh[-1,:] / Vh[-1,-1]\n    H_dlt = L.reshape(3, 3)\n    return H_dlt\nH_dlt = dlt_homography(src_pts, dst_pts)\ndlt_img = cv2.warpPerspective(img, H_dlt, (w,h))\nprint('15. Direct Linear Transformation (DLT):')\ncv2_imshow(dlt_img)"),
    ("Exp_16_CannyMethod", "gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\nedges_canny = cv2.Canny(gray_img, 100, 200)\nprint('16. Canny Edge Detection:')\ncv2_imshow(edges_canny)"),
    ("Exp_17_SobelX", "gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\nsobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)\nsobelx = cv2.convertScaleAbs(sobelx)\nprint('17. Sobel X:')\ncv2_imshow(sobelx)"),
    ("Exp_18_SobelY", "gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\nsobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)\nsobely = cv2.convertScaleAbs(sobely)\nprint('18. Sobel Y:')\ncv2_imshow(sobely)"),
    ("Exp_19_SobelXY", "gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\nsobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)\nsobelx = cv2.convertScaleAbs(sobelx)\nsobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)\nsobely = cv2.convertScaleAbs(sobely)\nsobelxy = cv2.bitwise_or(sobelx, sobely)\nprint('19. Sobel XY:')\ncv2_imshow(sobelxy)"),
    ("Exp_20_LaplacianNegativeCenter", "gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\nkernel_neg_center = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])\nlaplacian_neg = cv2.filter2D(gray_img, cv2.CV_64F, kernel_neg_center)\nsharpened_neg = np.clip(gray_img - laplacian_neg, 0, 255).astype(np.uint8)\nprint('20. Laplacian (Negative Center):')\ncv2_imshow(sharpened_neg)"),
    ("Exp_21_LaplacianExtendedDiagonals", "gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\nkernel_diag = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])\nlaplacian_diag = cv2.filter2D(gray_img, cv2.CV_64F, kernel_diag)\nsharpened_diag = np.clip(gray_img - laplacian_diag, 0, 255).astype(np.uint8)\nprint('21. Laplacian (Extended Diagonals):')\ncv2_imshow(sharpened_diag)"),
    ("Exp_22_LaplacianPositiveCenter", "gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\nkernel_pos_center = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])\nsharpened_pos = cv2.filter2D(gray_img, -1, kernel_pos_center)\nprint('22. Laplacian (Positive Center):')\ncv2_imshow(sharpened_pos)"),
    ("Exp_23_UnsharpMasking", "gaussian_blur = cv2.GaussianBlur(img, (9,9), 10.0)\nunsharp_img = cv2.addWeighted(img, 1.5, gaussian_blur, -0.5, 0)\nprint('23. Unsharp Masking:')\ncv2_imshow(unsharp_img)"),
    ("Exp_24_HighBoost", "gaussian_blur = cv2.GaussianBlur(img, (9,9), 10.0)\nA = 1.2\nhigh_boost_img = cv2.addWeighted(img, A, gaussian_blur, -1, 0)\nprint('24. High-Boost Masking:')\ncv2_imshow(high_boost_img)"),
    ("Exp_25_GradientMasking", "gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\nsobelx = cv2.convertScaleAbs(cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5))\nsobely = cv2.convertScaleAbs(cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5))\nsobel_mag = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)\nsobel_mag_bgr = cv2.cvtColor(sobel_mag, cv2.COLOR_GRAY2BGR)\ngradient_sharpened = cv2.addWeighted(img, 1.0, sobel_mag_bgr, 0.5, 0)\nprint('25. Gradient Masking:')\ncv2_imshow(gradient_sharpened)"),
    ("Exp_26_Watermarking", "watermark = np.zeros_like(img)\ncv2.putText(watermark, 'COLAB WATERMARK', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)\nwatermarked_img = cv2.addWeighted(img, 1.0, watermark, 0.3, 0)\nprint('26. Watermarked Image:')\ncv2_imshow(watermarked_img)"),
    ("Exp_27_CropCopyPaste", "cropped_region = img[100:300, 100:300]\nprint('27a. Cropped Region:')\ncv2_imshow(cropped_region)\npasted_img = img.copy()\npasted_img[0:200, 0:200] = cropped_region\nprint('27b. Pasted Image:')\ncv2_imshow(pasted_img)")
]

setup_code = [
    "import cv2\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from google.colab.patches import cv2_imshow\n",
    "import urllib.request\n",
    "\n",
    "# Download sample image\n",
    "opener = urllib.request.build_opener()\n",
    "opener.addheaders = [('User-agent', 'Mozilla/5.0')]\n",
    "urllib.request.install_opener(opener)\n",
    "urllib.request.urlretrieve('https://raw.githubusercontent.com/opencv/opencv/master/samples/data/building.jpg', 'building.jpg')\n",
    "img = cv2.imread('building.jpg')\n",
    "print('Original Image:')\n",
    "cv2_imshow(img)\n"
]

def create_notebook(filename, exp_code):
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": setup_code
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\\n" for line in exp_code.split("\\n")]
            }
        ],
        "metadata": {
            "colab": {
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open(filename, 'w') as f:
        json.dump(nb, f, indent=2)

for folder, code in experiments:
    path = os.path.join(base_dir, folder)
    os.makedirs(path, exist_ok=True)
    nb_path = os.path.join(path, f"{folder}.ipynb")
    create_notebook(nb_path, code)

print("Generated 27 folders and Colab notebooks.")
