import cv2 as cv
import numpy as np
import skimage
from skimage.morphology import skeletonize
from scipy.interpolate import make_interp_spline
import matplotlib as mpl
import matplotlib.cm



class Detectors(object):
    def __init__(self,roi_range,frame,name='None'):
        x, y, w, h = roi_range
        self.roi = np.array([x, y, x+w, y+h])

        self.roi_for_transform = None

        self.x_range = frame.shape[1]
        self.y_range = frame.shape[0]
        self.roi_size = max(w,h)

        self.b_update_roi = False

        self.key_no_sum = 5

        self.b_contour_max = True

        if name is None:
            self.name = ''
        else:
            self.name = name


        #porcine bronchus_out_1X_25_fps_out
        self.h_range = [2,255]
        self.s_range = [2,255]
        self.v_range = [0,109]
        self.morph_h, self.morph_v, self.morph_open = 3, 3 , 2
        self.contour_area_thresh = 0.7
        

        #V1-Complex terrains_colon
        # self.h_range = [0,255]
        # self.s_range = [0,255]
        # self.v_range = [0,50]
        # self.morph_h, self.morph_v, self.morph_open = 10, 1, 3
        # self.contour_area_thresh = 0.1

        self.area_thresh = 0.2
        self.b_first = True



        self.b_curve_r_l = True

        self.linewidth = 3
        self.linewidth_mesh = 5
        self.radius=10

        self.cmap = mpl.cm.get_cmap('viridis')

        self.linecolor=[0,255,0]
        self.circle_color = [0,255,0]

        self.count = 0

       

    def Detect_curve(self, frame):
        mask = self.segment(frame)
        keys_x, keys_y = self.find_key_points(mask,self.area_thresh,self.key_no_sum)
        keys_x, keys_y, x_new, y_new = self.make_interp(keys_x, keys_y)
        
        key_x = keys_x+self.roi_for_transform[0]
        key_y = (keys_y+self.roi_for_transform[1])

        x_new = x_new+self.roi_for_transform[0]
        y_new = (y_new+self.roi_for_transform[1])
        return key_x, key_y, x_new, y_new
        
    def segment(self,frame):

        self.roi_for_transform = self.roi
        x0,y0,x1,y1 = self.roi

        frame = frame[y0:y1,x0:x1,:]
        #cv.imshow("extract",frame)

        hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
        img_h = hsv[:,:,0].copy()
        img_s = hsv[:,:,1].copy()
        img_v = hsv[:,:,2].copy()

        mh = np.bitwise_and(img_h>=self.h_range[0],img_h<=self.h_range[1])
        ms = np.bitwise_and(img_s>=self.s_range[0],img_s<=self.s_range[1])
        mv = np.bitwise_and(img_v>=self.v_range[0],img_v<=self.v_range[1])

        maskh = np.zeros_like(img_v)
        maskh[mh]=255

        maskv = np.zeros_like(img_v)
        maskv[mv]=255


        se = cv.getStructuringElement(cv.MORPH_RECT, (self.morph_v, self.morph_v), (-1, -1))
        maskv = cv.morphologyEx(maskv, cv.MORPH_DILATE, se, iterations=1)
        
        mask_and = np.zeros_like(img_v)
        mask_and[maskv>0] = 255

        se = cv.getStructuringElement(cv.MORPH_RECT, (self.morph_open, self.morph_open), (-1, -1))
        mask_open = cv.morphologyEx(mask_and, cv.MORPH_OPEN, se, iterations=1)

        cv.imshow("mask"+self.name,mask_open)
        cv.waitKey(1)
        contours, hierarchy = cv.findContours(mask_open, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        index = 0
        max_area = 0

        for c in range(len(contours)):
            area = cv.contourArea(contours[c])
            if area > max_area:
                max_area = area
                index = c

        if self.b_first:
            self.b_first = False
            self.contour_area_thresh = max_area*self.contour_area_thresh

        return mask_open

    def find_end(self,skeleton):
        kernel = np.uint8([[1,  1, 1],
                        [1,  10, 1],
                        [1,  1, 1]])
        src_depth = -1
        skeleton[np.where(skeleton>0)] = 1
        filtered = cv.filter2D(skeleton, src_depth, kernel)
        endpoints = np.where(filtered == 11)
        endpoints = np.array([endpoints[1], endpoints[0]]).T#x,y
        return endpoints
    
    def find_skeleton(self,contour,mask,key_no=3):
        epsilon = 0.005 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour,epsilon,True)

        mask0 = np.zeros_like(mask)
        mask0=cv.drawContours(mask0, [approx], 0, (255), -1)

        skeleton_lee = skeletonize(mask0, method='lee')

        contours, hierarchy = cv.findContours(skeleton_lee, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        endpoints = self.find_end(skeleton_lee)

        if self.b_curve_r_l:
            endpoints_start = endpoints[np.argmin(endpoints[:,0])]
            endpoints_end = endpoints[np.argmax(endpoints[:,0])]

        else:
            endpoints_start = endpoints[np.argmin(endpoints[:,1])]
            endpoints_end = endpoints[np.argmax(endpoints[:,1])]

        line = contours[0].reshape(-1,2)
        

        indice_start = np.argmin((line-endpoints_start)[:,0]* (line-endpoints_start)[:,0]+(line-endpoints_start)[:,1]* (line-endpoints_start)[:,1])
        indice_end = np.argmin((line-endpoints_end)[:,0]* (line-endpoints_end)[:,0]+(line-endpoints_end)[:,1]* (line-endpoints_end)[:,1])

        if indice_start > indice_end:
            line = line[indice_start:indice_end+1:-1,:]
        else:
            line = line[indice_start:indice_end+1,:]

        indices = np.linspace(0,line.shape[0]-1,key_no)
        key_p = line[indices.astype(np.int)]

        key_x = key_p[:,0]
        key_y = key_p[:,1]

        return key_x, key_y

    def find_key_points(self,mask0,area_thresh=0.1,key_no_sum=6):
        if self.count == 840:
            x = 0

        contours, hierarchy = cv.findContours(mask0, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # areas = np.zeros(len(contours))
        # contour_centers = np.zeros((len(contours),2))
        # lengths = np.zeros(len(contours))
        # contour_indices = np.zeros(len(contours),dtype=np.int)
        # for c in range(len(contours)):
        #     area = cv.contourArea(contours[c])
        #     if area > self.contour_area_thresh:
        #         areas[c]= area
        #         lengths[c] = cv.arcLength(contours[c], True)

        #         x, y, w, h = cv.boundingRect(contours[c])
        #         contour_centers[c] = np.array([x+w//2,y+h//2])
        #         contour_indices[c] = c

        if not self.b_contour_max:

            lengths = []
            contour_centers = []
            contour_indices = []
            contours_filter = []
            for c in range(len(contours)):
                area = cv.contourArea(contours[c])
                if area > self.contour_area_thresh:

                    lengths.append(cv.arcLength(contours[c], True))

                    x, y, w, h = cv.boundingRect(contours[c])
                    contour_centers.append(np.array([x+w//2,y+h//2]))
                    contour_indices.append(c)
                    contours_filter.append(contours[c])
            
            contours = contours_filter

            lengths = np.array(lengths)
            contour_centers = np.asarray(contour_centers)
            contour_indices = np.arange(len(contours),dtype=np.int)

        else:
            lengths = []
            contour_centers = []
            contour_indices = []
            contours_filter = []

            index = 0
            max_area = 0
            for c in range(len(contours)):
                area = cv.contourArea(contours[c])
                if area > max_area:
                    max_area = area
                    index = c
            
            lengths.append(cv.arcLength(contours[index], True))
            x, y, w, h = cv.boundingRect(contours[index])
            contour_centers.append(np.array([x+w//2,y+h//2]))
            contour_indices.append(index)
            contours_filter.append(contours[index])
            
            contours = contours_filter

            lengths = np.array(lengths)
            contour_centers = np.asarray(contour_centers)
            contour_indices = np.arange(len(contours),dtype=np.int)
            




        #area_thresh = np.sum(areas)*area_thresh
        # contour_mask = areas>self.contour_area_thresh

        # contour_centers = contour_centers[contour_mask]
        # lengths = lengths[contour_mask]
        # contour_indices = contour_indices[contour_mask]

        for i in range(contour_indices.shape[0]):
            if i ==0:
                contour_concat = contours[contour_indices[i]]
            else:
                contour_concat = np.r_[contour_concat,contours[contour_indices[i]]]
            
        
        x, y, w, h = cv.boundingRect(contour_concat)
        
        #update roi
        if self.b_update_roi:
            self.roi = np.array([self.roi[0]+x+w//2-self.roi_size,
                        self.roi[1]+y+h//2-self.roi_size,
                        self.roi[0]+x+w//2+self.roi_size,
                        self.roi[1]+y+h//2+self.roi_size])
        else:
            self.roi = self.roi


        self.roi[self.roi<0]=0
        if self.roi[2]>self.x_range:
            self.roi[2]=self.x_range
        if self.roi[3]>self.y_range:
            self.roi[3] = self.y_range

        if w >h:
            self.b_curve_r_l = True
        else:
            self.b_curve_r_l = False

        if self.b_curve_r_l:
            contour_indices = np.argsort(contour_centers[:,0])
        else:
            contour_indices = np.argsort(contour_centers[:,1])


        length_sum=np.sum(lengths)
        key_nos = (lengths/length_sum*self.key_no_sum).astype(int)
        key_nos[key_nos==0]=1
        key_nos[-1]= self.key_no_sum-np.sum(key_nos[:-1])

        keys_x=np.zeros(key_no_sum)
        keys_y=np.zeros(key_no_sum)
        
        count=0
        b_first_key = True
        if contour_indices.shape[0]>1:
            xx =1

        for i in range(contour_indices.shape[0]):
            indice = contour_indices[i]
            key_no=key_nos[indice]
            key_x, key_y = self.find_skeleton(contours[indice],mask=mask0,key_no=key_no)
            if b_first_key:
                b_first_key = False
                keys_x[count:count+key_no]=key_x
                keys_y[count:count+key_no]=key_y

            else:
                x_diff1 = keys_x[count-1]-key_x[0]
                y_diff1 = keys_y[count-1]-key_y[0]
                diff1 = x_diff1**2+y_diff1**2

                x_diff2 = keys_x[count-1]-key_x[-1]
                y_diff2 = keys_y[count-1]-key_y[-1]
                diff2 = x_diff2**2+y_diff2**2  

                if diff1<diff2:
                    keys_x[count:count+key_no]=key_x
                    keys_y[count:count+key_no]=key_y
                else:
                    keys_x[count:count+key_no]=key_x[::-1]
                    keys_y[count:count+key_no]=key_y[::-1]

            count +=key_no

            self.count +=1

        return keys_x,keys_y
    

    def trans_keys(self,key_x, key_y, x_new, y_new):
        key_x = key_x+self.roi_for_transform[0]
        key_y = self.y_range-(key_y+self.roi_for_transform[1])

        x_new = x_new+self.roi_for_transform[0]
        y_new = self.y_range-(y_new+self.roi_for_transform[1])

        
        return key_x, key_y, x_new, y_new

    def make_interp(self,keys_x, keys_y):
        t = np.linspace(0,1,keys_x.shape[0])
        key_p = np.zeros((keys_x.shape[0],2))
        key_p[:,0]=keys_x
        key_p[:,1]=keys_y
        spl = make_interp_spline(t,key_p)

        t_new = np.linspace(0, 1, 100)
        x_new, y_new = spl(t_new).T

        return keys_x, keys_y, x_new, y_new

    def draw_curve(self, frame, key_x, key_y, x_new, y_new):
        #show
        mask0 = np.zeros_like(frame[:,:,0])
        mask0[y_new.astype(np.int),x_new.astype(np.int)]=255
        
        se = cv.getStructuringElement(cv.MORPH_RECT, (self.linewidth, self.linewidth), (-1, -1))
        binary = cv.morphologyEx((mask0).astype(np.uint8), cv.MORPH_DILATE, se)

        frame[np.where(binary>0)]=self.linecolor

        # mask0 = np.zeros_like(frame[:,:,0])
        # mask0[key_y.astype(np.int),key_x.astype(np.int)]=255
        
        # se = cv.getStructuringElement(cv.MORPH_RECT, (self.radius, self.radius), (-1, -1))
        # binary = cv.morphologyEx((mask0).astype(np.uint8), cv.MORPH_DILATE, se)

        # frame[np.where(binary>0)]=[0,0,255] 

        return frame



if __name__=='__main__':

    num = 4

    b_first = True
    detectors = []
    detectors_name = []

    for i in range(201):
        name = './1/'+str(i)+'.tiff'
        frame = cv.imread(name)
        
        if b_first:
            b_first = False
            for num_count in range(num):
                detectors_name.append(str(num_count))

                cv.namedWindow("ROI",cv.WINDOW_NORMAL)
                x, y, w, h = cv.selectROI("ROI", frame, True, False)
                cv.destroyAllWindows()

                detectors.append(Detectors([x, y, w, h],frame,str(num_count)))
        
            fourcc = cv.VideoWriter_fourcc(*'XVID')
            out = cv.VideoWriter('out.avi', fourcc, 10, (frame.shape[1], frame.shape[0]))

            

        cv.namedWindow('show')

        key_x_save = []
        key_y_save = []
        count = 0

        if frame is not None:
            frame_show = frame.copy()
            for j in range(num):

                keys_x, keys_y, x_new, y_new = detectors[j].Detect_curve(frame)

                key_x_s, key_y_s, x_new_s, y_new_s = detectors[j].trans_keys(keys_x, keys_y, x_new, y_new)

                key_x_save.append(key_x_s)
                key_y_save.append(key_y_s)

                np.savetxt(detectors_name[j]+'_x1.txt',np.asarray(key_x_save[j::num]).reshape(-1,detectors[j].key_no_sum))
                np.savetxt(detectors_name[j]+'_y1.txt',np.asarray(key_y_save[j::num]).reshape(-1,detectors[j].key_no_sum))

                frame_show = detectors[j].draw_curve(frame_show, keys_x, keys_y, x_new, y_new)
                #cv.rectangle(frame_show, (detectors[j].roi[0], detectors[j].roi[1]), (detectors[j].roi[2], detectors[j].roi[3]), (0, 255, 0), 6)#矩形
                
            out.write(frame_show)
            cv.imshow("show",frame_show)
            key = cv.waitKey(1)

            count += 1


            if key == 13:#press enter
                cv.destroyAllWindows()
                break

        else:
            cv.destroyAllWindows()
            break


