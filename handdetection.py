#Gerekli kütüphaneleri yükleme
import numpy as np
import cv2
import math

#Kamera açma
capture = cv2.VideoCapture(0)

while capture.isOpened():
    
    #Kameradan kareleri yakalıyoruz
    ret, frame = capture.read()
    
    #Dikdörtgen penceresinden el verilerini alıyoruz
    crop_image = frame[0:250, 0:250]
    
    #Gürültü gidermek için Gaussian blur uyguluyoruz
    blur = cv2.GaussianBlur(crop_image, (3,3), 0)
    
    # Görüntü rengini BGR-> HSV değiştiriyoruz
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    
    #Ten rengi ve geri kalanın siyah olacağı şekilde ikili görüntü oluşturuyoruz 
    mask2 = cv2.inRange(hsv, np.array([0,30,60]), np.array([40,255,255]))
    
    #Morfolojik dönüşüm
    kernel = np.ones((5,5))
    
    #Arka plan gürültüsünü filtrelemek için morfolojik dönüşümler uyguluyoruz
    dilation = cv2.dilate(mask2, kernel, iterations = 1)
    erosion = cv2.erode(dilation, kernel, iterations = 1)    
       
    #Gauss bulanıklığı ve eşik uyguluyourz
    filtered = cv2.GaussianBlur(erosion, (3,3), 0)
    ret,thresh = cv2.threshold(filtered, 127, 255, 0)
  
    #Konturleri buluyoruz
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
    
    try:
        #Maksimum alana sahip kontur buluyoruz
        contour = max(contours, key = lambda x: cv2.contourArea(x))
       #Bulduğumuz konturun (el görüntüsünü) çevreleyen dikdörtgeni oluşturuyoruz
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(crop_image,(x,y),(x+w,y+h),(0,255,0),2)
    
        #Konturun dış bukey gövdesini bulmak için bu fonksiyonu kullanıyoruz
        hull = cv2.convexHull(contour)
        #Konturu çiziyoruz
        drawing = np.zeros(crop_image.shape,np.uint8)
        cv2.drawContours(drawing,[contour],-1,(0,255,0),0)
        cv2.drawContours(drawing,[hull],-1,(0,0,255),0)
        
        # Dışbukeyin kusurlarını buluyoruz
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour,hull)
        
        # Başlangıç ve bitiş noktasından uzak noktanın açısını bulmak için cosinüs kuralını kullanıyoruz, 
        #yani tüm kusurlar için dışbükey noktalar (parmak uçları)
        count_defects = 0
        
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            angle = (math.acos((b**2 + c**2 - a**2)/(2*b*c))*180)/3.14
            
           
            cv2.line(crop_image,start,end,[0,255,0],2)
    
    except:
        pass
    cv2.imshow("Gesture", frame)
    all_image = np.hstack((drawing, crop_image))
    cv2.imshow('Contours', all_image)
   
      
    # Kamerayı kapatmak için q harfine basıyoruz
    if cv2.waitKey(2) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()






