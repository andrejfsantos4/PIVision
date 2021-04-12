import numpy as np
import cv2
import cv2.aruco as aruco
#import copy

def img_resize(img, *new_size):
    #Resizes the image img to fit size in new_size. 
    #If new_size only has 1 parameter, it is considered as the final height of img (maintaing aspect ratio)
    #Receives img as array of uint8 and new_size as tuple
    if type(new_size[0]) == int:
        dim = (round(new_size[0]/img.shape[0] * img.shape[1]), new_size[0])
    else:
        dim = (new_size[0][0], new_size[0][1])

    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return img

#Positions of Aruco codes corners, for template size 1654x2339, starting at (0,0) top left:
    #Top left Aruco:
        #Top left corner: (45,46)
        #Top Right corner: (281,46)
        #Bottom right corner: (281,282)
        #Bottom left corner: (45,282)
    #Top right Aruco:
        #Top left corner: (1367,46)
        #Top Right corner: (1603,46)
        #Bottom right corner: (1603,282)
        #Bottom left corner: (1367,282)
    #Middle bottom Aruco:
        #Top left corner: (748,2134)
        #Top Right corner: (905,2134)
        #Bottom right corner: (905,2289)
        #Bottom left corner: (748,2289)
aruco_pos = np.array([[[45,46],[281,46], [281,282], [45,282]],
                      [[1367,46],[1603,46],[1603,282],[1367,282]],
                      [[748,2134],[905,2134],[905,2289],[748,2289]]])

#%%Real time video from webcam
# Detect markers - Real-time video input
img_template = cv2.imread('Template.png')
video_in = cv2.VideoCapture(0, cv2.CAP_DSHOW) #From PC webcam
#video_in = cv2.VideoCapture("http://ip.xx.x.../video") #from smartphone camera using app such as DroidCam
#fps = video_in.get(cv2.CAP_PROP_FPS)
dict4_7by7 = aruco.custom_dictionary(4, 7)  #create dictionary
arucoParameters = aruco.DetectorParameters_create()
src_pts = np.zeros((4*3,2)) #source points: corners in image
dst_pts = np.zeros((4*3,2)) #destination points: corners in template
counter = 0
while(True):
    counter += 1
    ret = video_in.grab()
    if counter%4 == 0: #This is used to discard some frames, thereby reducing the frame rate
        ret, frame = video_in.retrieve()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_frame, dict4_7by7, parameters=arucoParameters)
        if np.all(ids != None): #if at least one aruco code has been detected
            for i in range(len(ids)):
                for j in range(4):
                    src_pts[4*i+j,:] = corners[i][0][j]
                    dst_pts[4*i+j,:] = aruco_pos[ids[i,0],j,:]
            H, mask = cv2.findHomography(src_pts[0:4*len(ids),:], dst_pts[0:4*len(ids),:], cv2.RANSAC, 4)
            img_transf = cv2.warpPerspective(frame, H, (img_template.shape[1],img_template.shape[0]))
            img_transf = img_resize(img_transf, 640)
        else: #otherwise just show the video frame
            img_transf = frame
        cv2.imshow('Display', img_transf)
        if cv2.waitKey(1) & 0xFF == ord('q'): #detects key 'q' for quiting
            break

video_in.release()
cv2.destroyAllWindows()

#%%Single image
# img_template = cv2.imread('Template.png')
# img = cv2.imread('Test3.jpg') 
# img = img_resize(img, 1080)
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# dict4_7by7 = aruco.custom_dictionary(4, 7)  #create dictionary
# arucoParameters = aruco.DetectorParameters_create()

# corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_img, dict4_7by7, parameters=arucoParameters)
# if np.all(ids != None): #if at least one aruco code has been detected
#     img_detected = copy.deepcopy(img) 
#     aruco.drawDetectedMarkers(img_detected, corners, ids) #overlay corners and ids
# else:
#     print('No Aruco codes detected, terminating.')
#     exit()
# cv2.imshow('image',img_detected)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# src_pts = np.zeros((4*len(ids),2)) #source points: corners in image
# dst_pts = np.zeros((4*len(ids),2)) #destination points: corners in template
# for i in range(len(ids)):
#     for j in range(4):
#         src_pts[4*i+j,:] = corners[i][0][j]
#         dst_pts[4*i+j,:] = aruco_pos[ids[i,0],j,:]
# H, mask = cv2.findHomography(src_pts, dst_pts)#, cv2.RANSAC, 4)
# img_transf = cv2.warpPerspective(img, H, (img_template.shape[1],img_template.shape[0]))

# img_transf = img_resize(img_transf, 720)

# cv2.imshow('Transformed',img_transf)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
