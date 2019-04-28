# ======================================================================================================================================================================= #
#-------------> Project 02 <---------------#
# ======================================================================================================================================================================= #
# Course    :-> ENPM673 - Perception for Autonomous Robots
# Date      :-> 13 March 2019
# Authors   :-> Siddhesh Rane, Niket Shah, Sudharshan B.
# ======================================================================================================================================================================= #

# ======================================================================================================================================================================= #
# Import Section for Importing library
# ======================================================================================================================================================================= #
import cv2
import numpy as np
import matplotlib.pyplot as plt
import  math, copy

# ======================================================================================================================================================================= #
# Undistort the frame using given camera and distortion matrix
# ======================================================================================================================================================================= #
def undistort_image(img):
    mtx = np.array([[1.15422732e+03,0.00000000e+00,6.71627794e+02],[0.00000000e+00,1.14818221e+03,3.86046312e+02],[0.00000000e+00,0.00000000e+00,1.00000000e+00]])
    dist =  np.array([[-2.42565104e-01,-4.77893070e-02,-1.31388084e-03,-8.79107779e-05,2.20573263e-02]])
    dst = cv2.undistort(frame,mtx, dist, None, mtx)
    return dst

# ======================================================================================================================================================================= #
# Color Space Conversion Section
# ======================================================================================================================================================================= #
#  BGR to HSV Colorspace
def get_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv

# ======================================================================================================================================================================= #
# BGR to HSL Colorspace
# ======================================================================================================================================================================= #
def get_hsl(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    return hls

# ======================================================================================================================================================================= #
# BGR to LAB Colorspace
# ======================================================================================================================================================================= #
def get_lab(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    return lab

# ======================================================================================================================================================================= #
# Function to perform Histogram Equalisation on the each channel of a given image
# ======================================================================================================================================================================= #
def histogram_eq(img):
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    return img

# ======================================================================================================================================================================= #
# Function to split channels of a given image
# ======================================================================================================================================================================= #
def get_channel(img, channel_no=0):
    channel = img[:,:,channel_no]
    return channel

# ======================================================================================================================================================================= #
# Function to Display Image
# ======================================================================================================================================================================= #
def disp(title,img):
    cv2.imshow(title,img)

# ======================================================================================================================================================================= #
# Function to get yellow region mask in a given Image
# ======================================================================================================================================================================= #
def get_yellow_mask(hsv):
    yellow_hsv_low  = np.array([ 0, 85, 111])
    yellow_hsv_high = np.array([ 40, 255, 255])
    yellow_mask = cv2.inRange(hsv, yellow_hsv_low, yellow_hsv_high)
    return yellow_mask
    #ylw_hsv = cv2.bitwise_and(img,img,mask=mask)

# ======================================================================================================================================================================= #
# Function to get white region mask in a given Image
# ======================================================================================================================================================================= #
def get_white_mask(hsv):
    white_hsv_low  = np.array([ 0, 0, 176])
    white_hsv_high = np.array([ 255, 80, 255])
    white_mask = cv2.inRange(hsv, white_hsv_low, white_hsv_high)
    return white_mask

# ======================================================================================================================================================================= #
# Function to get combined mask of yellow and white region in a given Image
# ======================================================================================================================================================================= #
def get_whilte_yellow(img,hsv):
    yellow_hsv_low  = np.array([ 0, 85, 111])
    yellow_hsv_high = np.array([ 40, 255, 255])
    mask = cv2.inRange(hsv, yellow_hsv_low, yellow_hsv_high)
    ylw_hsv = cv2.bitwise_and(img,img,mask=mask)
    white_hsv_low  = np.array([ 0, 0, 176])
    white_hsv_high = np.array([ 255, 80, 255])
    mask = cv2.inRange(hsv, white_hsv_low, white_hsv_high)
    wht_hsv = cv2.bitwise_and(img,img,mask=mask)
    yplusw = cv2.bitwise_or(wht_hsv,ylw_hsv)
    gray = cv2.cvtColor(yplusw,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    ret,thres = cv2.threshold(blur,127,255,cv2.THRESH_BINARY)
    return thres

# ======================================================================================================================================================================= #
# Function to Compute Homographgy by performinng perspective transform and get the top view 
# ======================================================================================================================================================================= #
def compute_homography(img):
    #pts1 = np.float32([[550,500],[850,500],[1250,730],[150,730]])
    pts1 = np.float32([[550,450], [750,450], [1150,600], [150,600]])
    width = 800
    height = 600
    #pts2 = np.float32([[0,0],[500,0],[500,890],[0,890]])
    pts2 = np.float32([[0,0], [width-1,0], [width-1, height-1], [0, height-1]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    result = cv2.warpPerspective(img,M,(width,height))
    return result , M

# ======================================================================================================================================================================= #
# Function to get thresholded image for current channel
# ======================================================================================================================================================================= #
def get_color_thres(img, thres=(0,255),channel_no=0,img_type='hls_s'):
    img_channel = img[:,:,channel_no]
    if img_type == 'hls_l':
        # normalize
        img_channel = img_channel * (255/np.max(img_channel))
    if img_type == 'lab_b':
        if np.max(img_channel) > 175:
            img_channel = img_channel * (255/np.max(img_channel))        
    thres_img = np.zeros_like(img_channel)
    thres_img[(img_channel > thres[0]) & (img_channel <= thres[1])] = 1
    return thres_img

# ======================================================================================================================================================================= #
# Function to get sobelX thresholded image for current channel
# ======================================================================================================================================================================= #
def getSobelX(img, kernel=3, thres = (0,255)):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel)
    sobel_x_abs = np.absolute(sobel_x)
    scaled = np.uint8(255*1.0*sobel_x_abs/np.max(sobel_x_abs))
    sobel_thres = np.zeros_like(scaled)
    sobel_thres[(scaled >= thres[0]) & (scaled < thres[1])] = 1
    return sobel_thres

# ======================================================================================================================================================================= #
# Function to get sobelY thresholded image for current channel
# ======================================================================================================================================================================= #
def getSobelY(img, kernel=3, thres = (0,255)):
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel)
    sobel_y_abs = np.absolute(sobel_y)
    scaled = np.uint8(255*1.0*sobel_y_abs/np.max(sobel_y_abs))
    sobel_thres = np.zeros_like(scaled)
    sobel_thres[(scaled >= thres[0]) & (scaled < thres[1])] = 1
    return sobel_thres

# ======================================================================================================================================================================= #
# Function to get sobelMag thresholded image for current channel
# ======================================================================================================================================================================= #
def getSobelMag(img, kernel=7, thres = (0,255)):
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #sobelx = cv2.Sobel(gray,cv2.CV_64F, 1, 0, ksize=kernel)
    #sobely = cv2.Sobel(gray,cv2.CV_64F, 0, 1, ksize=kernel)
    sobelx = getSobelX(img,kernel,thres)
    sobely = getSobelY(img,kernel,thres)
    sobel_mag = np.sqrt(np.square(sobelx) + np.square(sobely))
    scaled = np.uint8(255*1.0*sobel_mag/np.max(sobel_mag))
    mag_thres = np.zeros_like(scaled)
    mag_thres[(scaled >= thres[0]) & (scaled < thres[1])] = 1
    return mag_thres

# ======================================================================================================================================================================= #
# Function to get sobelDir thresholded image for current channel
# ======================================================================================================================================================================= #
def getSobelDir(img, kernel=7, thres = (0,255)):
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #sobelx = cv2.Sobel(gray,cv2.CV_64F, 1, 0, ksize=kernel)
    #sobely = cv2.Sobel(gray,cv2.CV_64F, 0, 1, ksize=kernel)
    sobelx = getSobelX(img,kernel,thres)
    sobely = getSobelY(img,kernel,thres)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    sob_dir = np.arctan2(abs_sobely, abs_sobelx)
    scaled = np.uint8(255*1.0*sob_dir/np.max(sob_dir))
    dir_thres = np.zeros_like(scaled)
    dir_thres[(scaled >= thres[0]) & (scaled < thres[1])] = 1
    return dir_thres

# ======================================================================================================================================================================= #
# Function to perform histogram polyfit
# ======================================================================================================================================================================= #
def historgram_polyfit(warped_img, nwindows = 7, window_size = 90, minpix = 50):
    # First take a histogram of the bottom half of the image
    histogram = np.sum(warped_img[int(warped_img.shape[0]/2):,:], axis=0)
    # Starting from midpoint, find peaks of the left and right lanes 
    # of the histogram.
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # Adjust height of the window according to image size and number of windows.
    window_height = np.int(warped_img.shape[0]/nwindows)
    # Get all the non zero pixels on the image and separate them into x and y
    nonzero = warped_img.nonzero()
    nonzerox = np.array(nonzero[1])
    nonzeroy = np.array(nonzero[0])
    # Set current window position at the bottom of the histogram
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped_img.shape[0] - (window+1)*window_height
        win_y_high = warped_img.shape[0] - window*window_height
        win_xleft_low = leftx_current - window_size
        win_xleft_high = leftx_current + window_size
        win_xright_low = rightx_current - window_size
        win_xright_high = rightx_current + window_size
        # Get non zero pixels which are within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If number of non zero pixels is greater than the selecte threshold, 
        # recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    # the arrays of the indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    # This is a second order polynomial with coefficiens a, b , c
    if len(leftx) != 0 and leftx.any() != None and len(lefty) != 0 and lefty.any() != None:
        left_fit = np.polyfit(lefty, leftx, 2)
    else:
        left_fit = np.array([0,0,0])
    if len(rightx) != 0 and rightx.any() != None and len(righty) != 0 and righty.any() != None:
        right_fit = np.polyfit(righty, rightx, 2)
    else:
        right_fit = np.array([0,0,0])
    return left_fit, right_fit

    
# ======================================================================================================================================================================= #
# Function to compute the radius of curvature
# ======================================================================================================================================================================= #
def predict_turn(warped_img, left_fit, right_fit):
    # first get a conversion factor from pixels to meters.
    # in usa roads have width of 12ft = 3.7meters
    xm_per_pix = 3.7/300
    if left_fit.any() == None or right_fit.any() == None:
        return 0 , 0 , Nan
    # get a y coordinate in the top region of the image which will be used
    # to find two points on the left lane and right lane
    y_top = 1
    
    #left_lane and right lane x coordinates for y_top in pixels
    left_top_x = (left_fit[0]*y_top)**2 + left_fit[0]*y_top + left_fit[2]
    right_top_x = (right_fit[0]*y_top)**2 + right_fit[0]*y_top + right_fit[2]
    
    # Assume that the camera is located at the center of the car
    camera_center = 400
    
    # Lane center as mid of left and right lane bottom                        
    lane_center_x = (left_top_x + right_top_x)/2.
    # get center location in pixels
    center_px = (lane_center_x - camera_center)
    
    #Convert to meters
    center_meters = center_px * xm_per_pix
    
    turn = ""
    if center_meters  < -1.5:
        turn = "left"
    elif center_meters  >= -1 and center_meters < 0.9:
        turn = "straight"
    else:
        turn = "right"
    # Now our radius of curvature is in meters
    return turn

# ======================================================================================================================================================================= #
# Function to perform inverse warpping to superimpose the lane path back into image
# ======================================================================================================================================================================= #
def inverse_warp(undist, warped_img, left_fit, right_fit, M):
    # get image height. So we get y values from 0 to img height-1
    img_height = warped_img.shape[0]
    # get list of all y coordinates
    y_coord = np.linspace(0, img_height-1, img_height)
    
    # Get x coordinates for left and right lanes using polyfit values
    left_x = left_fit[0]*y_coord**2 + left_fit[1]*y_coord + left_fit[2]
    right_x = right_fit[0]*y_coord**2 + right_fit[1]*y_coord + right_fit[2]

    # create a numpy array of warped_img size so we can draw on it
    warp_zero = np.zeros_like(warped_img).astype(np.uint8)
    # make it three channel
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # use y and y coordinates of each line to get point format
    pts_left = np.array([np.transpose(np.vstack([left_x, y_coord]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_x, y_coord])))])
    points = np.hstack((pts_left, pts_right))

    # Draw the lane on warped image
    cv2.fillPoly(color_warp, np.int_([points]), (0,255, 0))
    # Calculate M inverse so we can warp this region on world frame
    Minv = np.linalg.inv(M)
    new_warp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
    # Combine with original frame
    unwarped = cv2.addWeighted(undist, 1, new_warp, 0.4, 0)
    return unwarped

# ======================================================================================================================================================================= #
# Function to overlay the text on image frame
# ======================================================================================================================================================================= #
def display_text(unwarped, turn):
    cv2.putText(unwarped, 'Turning direction: {}'.format(turn), (50, 150), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2) 


# Main Loop
vid = cv2.VideoCapture("project_video.mp4")
while True:
    ret, frame = vid.read()
    if ret == None or ret == False:
        break
        
    # Undistort the image
    frame = undistort_image(frame)
    # Get hsl
    hls = get_hsl(frame)
    hls = histogram_eq(hls)
    # Get hsv
    hsv = get_hsv(frame)
    # Get Lab
    lab = get_lab(frame)
    lab = histogram_eq(lab)
    
    '''
    In this we are only using hls-l and lab-b channel
    ''' 
    # get thresholded image for hls l channel
    img_thr_hls_l = get_color_thres(hls, thres=(200,255),channel_no=1,img_type='hls_l')
    # get thresholded image for lab b channel
    img_thr_lab_b = get_color_thres(lab, thres=(250,255),channel_no=2,img_type='lab_b')
    # combine hls-l with lab-b
    combined_l_b = np.zeros_like(img_thr_hls_l)
    combined_l_b[(img_thr_hls_l == 1) | (img_thr_lab_b == 1)] = 1
    homographic_img, M = compute_homography(combined_l_b)
    
    left_fit, right_fit = historgram_polyfit(homographic_img) 
    turn = predict_turn(combined_l_b, left_fit, right_fit)
    unwarped = inverse_warp(frame, homographic_img, left_fit, right_fit, M)
    display_text(unwarped, turn)
    # display frames
    #disp("warped",combined_l_b)
    #disp("yellow white",combined_w_y)
    disp("result",homographic_img)
    disp("frame",unwarped)
    
    key = cv2.waitKey(1)
    if key == 27:
        break
vid.release()
cv2.destroyAllWindows()
