# Advanced-Lane-Line-Detection-Using-Python-and-OpenCV
This project is aimed to do simple 'Lane Line Detection' to mimic 'Lane Departure Warning Systems' used in Self Driving Cars

## Step 1: Input Preparation:
Before initiating the image processing, lens distortions of the camera were removed by using the camera matrix as well as the distortion parameters that are provided to us. For this purpose, inbuilt cv2 function undistort() is utilized.
For denoising of the image, initially, Gaussian blur filter was used. However, it was observed that it does not provide any significant improvements in the quality of the video frame. Also, it slowed down the video processing speed. Hence, it was decided to skip the gaussian blurring filter.
 Next, to create a thresholded binary mask, Canny edge detection, as well as various combinations of the color channels, were tried. Outputs obtained using these masks are given below:
* <b> Sobel-x and Sobel-y mask
  <p>
    <img src="images/1.PNG">
  </p>
  First, the image frame was converted into a grayscale image and then binary thresholding was performed. Then Sobel x and Sobel y filter was applied to this thresholded image to extract the edges. Also, another mask was created by performing bitwise or operation on sobel_x and sobel_y.
* <b> hsv_yellow and hsv_white mask:
  <p>
    <img src="images/2.PNG">
  </p>
  First, the image frame was converted into the HSV format so we can use color segmentation on it. Since it is known that the lane lines have either yellow or white colors, two masks were created using lower and upper threshold values of yellow and white respectively. Thus hsv_yellow mask was able to distinguish the yellow lane line and the hsv_white mask was able to distinguish the white lane line. Finally to get the complete lane detection both the masks were combined by performing a binary_or operation on them. The resulting mask ‘hsv_y + hsv_w’ is shown above.
* <b> sobel_mag and sobel_direction mask:
  <p>
    <img src="images/3.PNG">
  </p>
  Sobel magnitude maks was created by computing the absolute magnitude of the sobel_x and sobel_y values. Similarly, Sobel direction mask was created by taking inverse tangent of the sobel_y/sobel_x values. Furthermore, both sobel_mag and sobel_dir were combined using bitwise_or operation.
* <b> lab_b and hls_l masks:
  <p>
    <img src="images/4.PNG">
  </p>
  After separating the image into  various spaces, it was observed that the B channel in the LAB color space produces the brightest output for yellow color and L channel in the HLS color space produces the brightest output for white color. Hence the image frame was first converted into HLS and lab color spaces and from that l-channel and b-channel was extracted. To improve the quality of these color channels, histogram equalization was performed on them and appropriate thresholding values were selected. Finally, both hls_l and lab_b were combined by using binary or operation to get complete lane lines.
  
##  Observations:
