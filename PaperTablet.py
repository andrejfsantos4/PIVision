
#TODO: Automatizar a escolha do treshold do erro, por exemplo 1/10 da largura da página
#TODO: Field for dict size and change in code 4->dict-size
#IDEA: subtract cropped image with transformed image to get a measure of homography quality.
#IDEA: In position error function detect if one or more corners are far away from their correct
#      positions, meaning that the homography is bad

import numpy as np
import cv2
from cv2 import aruco as aruco

def initArucoPos(template, aruco_dict, arucoParameters):
    #Returns the corners of the Aruco markers in the template image
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_template, aruco_dict, parameters=arucoParameters)
    #TODO test this
    if np.all(ids==None):
        print("The server was unable to detect Aruco markers in the template.")
        exit(1)
        
    aruco_pos = np.zeros((len(ids),4,2))
    aruco_pos[ids]=corners
    return aruco_pos

def ImgResize(img, *new_size):
    #Resizes the image img to fit size in new_size.
    #If new_size only has 1 parameter, it is considered as the final height of img (maintaing aspect ratio)
    #Receives img as array of uint8 and new_size as tuple
    if type(new_size[0]) == int:
        dim = (round(new_size[0]/img.shape[0] * img.shape[1]), new_size[0])
    else:
        dim = (new_size[0][0], new_size[0][1])
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return img

def GetHomography(frame, aruco_pos):
    #Detects Aruco codes in frame and estimates homography from frame to template.
    #If no Aruco codes are detected returns -1, otherwise returns the homography
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_frame, dict4_7by7, parameters=arucoParameters)

    if np.all(ids != None):  # if at least one aruco code has been detected
        src_pts = np.zeros((4 * len(ids), 2))  # source points: corners in image
        dst_pts = np.zeros((4 * len(ids), 2))  # destination points: corners in template
        for i in range(len(ids)):
            for j in range(4):
                src_pts[4 * i + j, :] = corners[i][0][j]
                dst_pts[4 * i + j, :] = aruco_pos[ids[i, 0], j, :]
        H, mask = cv2.findHomography(src_pts[0:4 * len(ids), :], dst_pts[0:4 * len(ids), :], cv2.RANSAC, 4)
    else:
        return -1
    return H

def ApplyDetect(img_template, frame, H, dict4_7by7, arucoParameters):
    #Applies the homography H to the frame and resizes it.
    #Then detects Aruco codes in the resized frame
    #Returns the transformed frame, the corners of Aruco codes and their ids
    img_transf = cv2.warpPerspective(frame, H, (img_template.shape[1],img_template.shape[0]))
    img_transf = ImgResize(img_transf, 800)
    gray_img_transf = cv2.cvtColor(img_transf, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_img_transf, dict4_7by7, parameters=arucoParameters)
    return img_transf, corners, ids

def PositionError(corners, ids, aruco_pos, resize_ratio):
    #Computes the average error, in pixels, between the Aruco codes positions in two different images
    #resize_ratio is the ratio between the width of the frame and the width of the template
    #Return -1 as flag if no Aruco codes have been detected
    if not isinstance(ids, np.ndarray): #means that ids is empty, i.e., no Aruco codes detected
        return -1

    ave_error = 0
    for i in range(len(ids)):
        ave_error = ave_error + np.sum(np.linalg.norm( (np.around(aruco_pos[ids[i]]*resize_ratio)-corners[i])[0] , axis=1))
    return ave_error/(4*len(ids))

def calling_function(img_in):
    #This function the "main" function which will call all other functions and implements specific protocols for each situation
    global H, aruco_pos, img_template, arucoParameters
    nparr = np.frombuffer(img_in, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if not isinstance(H,np.ndarray): #If no homography has been calculated yet
        H = GetHomography(frame, aruco_pos)
        if not isinstance(H,np.ndarray): #Unable to calculate homography
            img_transf = frame
            return cv2.imencode('.png', img_transf)[1].tobytes()

    #Apply the previous homography to the new frame and detect Aruco codes
    img_transf, corners, ids = ApplyDetect(img_template, frame, H, dict4_7by7, arucoParameters)
    average_error = PositionError(corners, ids, aruco_pos, img_transf.shape[0]/img_template.shape[0] )
    
    #!!!!!!!!!!!!!!!!!!!!!--------------------------------------------------------------
    if (not np.all(ids != None)) or average_error > 30: #If no aruco code has been detected OR if error is to large,
                                                         #means that there was motion and new H is required
        H_new = GetHomography(frame, aruco_pos)
        if not isinstance(H_new,np.ndarray): #Unable to calculate homography
            return cv2.imencode('.png', img_transf)[1].tobytes()

        img_transf_new, corners, ids = ApplyDetect(img_template, frame, H_new, dict4_7by7, arucoParameters)

        if np.all(ids != None): #If Aruco codes are detected with new homography, evaluate its error
            average_error = PositionError(corners, ids, aruco_pos, img_transf.shape[0]/img_template.shape[0] )
            if average_error < 30: #!!!!!!!!!!!!!!!!!!!!!--------------------------------------------------------------
                H = H_new
                img_transf = img_transf_new
            #else:
                #Bad homography so just show frame with original homography (which is the default behaviour)
                #TODO check if this is the best option, or maybe find an alternative way to compute a new H that produces better results
    #else:
        #Means that aruco codes were detected and error is small so show frame with original homography
    return cv2.imencode('.png', img_transf)[1].tobytes()

#Global variables:
img_template = cv2.imread('Template.png')
if img_template is None:
    print("The server was unable to read the template.")
    exit()
dict4_7by7 = aruco.custom_dictionary(4, 7)
arucoParameters = aruco.DetectorParameters_create()
aruco_pos = initArucoPos(img_template, dict4_7by7, arucoParameters)
H = -1
