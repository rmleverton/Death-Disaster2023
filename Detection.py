from ultralytics import YOLO
import cv2
import PIL
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops


model =  YOLO('yolov8x-seg.pt')

def detect(image):
    results = model(image)
    
    for result in results:
        detection_count = result.boxes.shape[0]
        #plotted = result.plot(conf=False, line_width=None, font_size=None, font='Arial.ttf', pil=True, img=None, im_gpu=None, kpt_radius=5, kpt_line=True, labels=True, boxes=True, masks=False, probs=False) 


        outmask = np.zeros((448, 640, 3))
        objects = []
        for i in range(detection_count):
            mask_raw = result.masks[i].cpu().data.numpy().transpose(1, 2, 0)
    
            # Convert single channel grayscale to 3 channel image
            mask_3channel = cv2.merge((mask_raw,mask_raw,mask_raw))

            # Get the size of the original image (height, width, channels)
            h2, w2, c2 = result.orig_img.shape
            
            # Resize the mask to the same size as the image (can probably be removed if image is the same size as the model)
            mask = cv2.resize(mask_3channel, (w2, h2))

            # Convert BGR to HSV
            hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)

            # Define range of brightness in HSV
            lower_black = np.array([0,0,0])
            upper_black = np.array([0,0,1])

            # Create a mask. Threshold the HSV image to get everything black
            mask = cv2.inRange(mask, lower_black, upper_black)

            # Invert the mask to get everything but black
            mask = cv2.bitwise_not(mask)
            #print(mask)

            if i == 0:
                outmask = mask
            
            else:
                #cv2.bitwise_or(outmask, mask)
                outmask += mask

            # # Apply the mask to the original image
            # masked = cv2.bitwise_and(result.orig_img, result.orig_img, mask=mask)
        
            cls = int(result.boxes.cls[i].item())
            name = result.names[cls]

        #     confidence = float(result.boxes.conf[i].item())

        #     #print(name + " : " + str(confidence))
            bounding_box = result.boxes.xyxyn[i]
            centre_X = (bounding_box[0].item() +bounding_box[2].item())/2
            centre_Y = (bounding_box[1].item() +bounding_box[3].item())/2

            object = [name, centre_X, centre_Y]
            objects.append(object)

    return outmask, objects#results, plotted

    #print(results)


if __name__ =="__main__":
    

    for filename in os.listdir("images"):
        imname = filename.split('.')[0]
        #detections, plotted = detect("images/" + filename)
        #cv2.imwrite("outputs/" + imname + ".jpg", plotted)
        mask, objects =  detect("images/" + filename)
        #print(objects)

        
        cv2.imwrite("outputs/" + imname + ".jpg", mask)

        img_w = 2880
        img_h = 1920
        txt = Image.new("RGBA",(img_w, img_h), (0, 0, 0, 255))
        edge = Image.new("RGBA",(img_w, img_h), (255, 255, 255, 0))
        bg = Image.new("RGBA",(img_w, img_h), (10, 10, 10, 255))
        m = Image.fromarray(mask)

        fnt = ImageFont.truetype("fonts/Domine-Bold.ttf", 60)

        d = ImageDraw.Draw(txt)

        for object in objects:
            #print(object)
            x = int(object[1] * img_w)
            
            y = int(object[2] * img_h)
            name = object[0]
            d.text((x,y), name, fill=(255, 255, 255, 255),font=fnt, align = "right")
        
        buf_m = np.asarray(m)
        grey_mask = m.convert("L")
        buf_mask = np.asarray(grey_mask)
        buf_mask = buf_mask * 0.02


        grey_txt = txt.convert("L")
        
        buf_t = np.asarray(grey_txt)

        grey_edge = edge.convert("L")
        buf_e = np.asarray(grey_edge)

        final = buf_m - buf_e
        final = final + buf_mask#buf_t

        btmask = np.where(np.logical_and(final>200, buf_t > 200), 0, 255)

        final = np.where(np.logical_and(final<255, buf_t == 255), 255, final)#final + buf_t

        btf = np.where(btmask==0, 0, final)

        # bt = np.where(np.logical_and(final>200, buf_t > 200), 255, final)
        # wt = np.where(np.logical_and(final != 255, buf_t == 255), 0, final)
        #a_mask = np.invert(final)
        #final = np.abs(final - buf_t)
        #result = np.where(final == 255,255,np.minimum(final,buf_t))
        #a_mask = a_mask - buf_t
        # nmask = final < 0
        # final[mask] = 255
        #final = Image.fromarray(final)
        #final = final.convert("RGBA")
        # print(final.size)
        #final.show()

        cv2.imwrite("outputs/" + imname + ".jpg", btf)
        #a_mask = ImageChops.invert(final.convert("L"))
        # a_mask = final.point(lambda p:p>0, '1')

        # alpha = Image.composite(final, bg, a_mask)
        # alpha.show()
        
        #alpha = Image.alpha_composite(final, txt)
        #alpha.show()
        # final += buf_t
        # final = Image.fromarray(final)
        # final.show()