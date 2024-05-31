import cv2
import numpy as np
import matplotlib.pyplot as plt
def dumm():
    pass
cam=cv2.VideoCapture(0)
def process(img):
    imgg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgb=cv2.GaussianBlur(imgg,(3,3),1)
    imgc=cv2.Canny(imgb,150,200)
    kernel=np.ones((3,3))
    imgd=cv2.dilate(imgc,kernel,iterations=2)
    imgt=cv2.erode(imgd,kernel,iterations=1)
    return imgt
def getContours(img):
    contours,hist=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cnt=[]
    bigArea=0
    for i in contours:
        area=cv2.contourArea(i)
        peri=cv2.arcLength(i,True)
        edges=cv2.approxPolyDP(i,0.02*peri,True)
        if len(edges)==4 and area>bigArea:
          cnt=edges
          bigArea=area
    return cnt
def reorder(myPoints): #takes array of type [[[]],[[]],[[]],...]
    myPoints=myPoints.reshape((4,2))
    myPointsNew=np.zeros((4,1,2),np.int32)
    add=myPoints.sum(1)
    myPointsNew[0]=myPoints[np.argmin(add)]
    myPointsNew[3]=myPoints[np.argmax(add)]
    diff=np.diff(myPoints,axis=1)
    myPointsNew[1]=myPoints[np.argmin(diff)]
    myPointsNew[2]=myPoints[np.argmax(diff)]
    return myPointsNew
    
def getWarp(img,cnt):
    if len(cnt)==0:
        return[]
    h,w=img.shape[:2]
    pts1=np.float32(reorder(cnt))
    pts2=np.float32([[0,0],[480,0],[0,640],[480,640]])
    matrix=cv2.getPerspectiveTransform(pts1,pts2)
    imgOutput=cv2.warpPerspective(img,matrix,(480,640))
    return imgOutput
while 1:
    d,img=cam.read()
    imgContour=img.copy()
    cnt=getContours(process(imgContour))
    cv2.drawContours(imgContour,cnt,-1,(0,250,0),20)
    cv2.imshow("lkh",imgContour)
    imgOut=getWarp(img.copy(),np.array(cnt))
    if len(imgOut)>0:
        imgCrop=imgOut[20:imgOut.shape[0]-20,20:imgOut.shape[1]-20]
        cv2.imshow('final',imgCrop)
        if cv2.waitKey(50)==ord('s'):
            cv2.imwrite("newPhoto.png",imgCrop)
    if cv2.waitKey(1)==ord('q'):
        break
cv2.destroyAllWindows()
del cam

    

