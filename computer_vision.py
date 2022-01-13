import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
import pickle as pkl 
import apriltag


class Webcam : 
    
    def __init__(self, port) :
        
        """
        Goal:   Handling communication with the camera 
        Input:  camera port 
        Output: camera frame
        """
        
        self.cap = cv2.VideoCapture(port)
        self.cap.set(3,1280)
        self.cap.set(4, 920)
        
    def get_frame(self, display=False) : 
        """
        Goal: Get the frames of the webcam
        Input: display = boolean with true if we want to display data as an image
        """
        ret, frame = self.cap.read()
        if ret : 
            frame = frame[:, :, ::-1]
            if display : 
                plt.imshow(frame)
            return frame
        else : 
            raise Exception('Webcam Stream Fail')
        
    def release(self) : 
        """
        Goal: release the webcam resource
        """
        self.cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
     

    
class ComputerVision : 
    
    def __init__(self, black_th, red_th, green_th, field_size) :
        
        """
        Goal:  Perform computer vision tasks   
        Input:  - threshold values for black, red and green
                - field size 
        Output: ComputerVision class used for computer vision related tasks 
        """
        
        # Basic Fields 
        self.field_size = field_size
        self.black_th = black_th
        self.red_th   = red_th 
        self.green_th = green_th
        
        # Cropping Positions 
        self.crop = None 

        
    def detection(self, image) : 
        
        """
        Goal:  Transformation of the image to fit the thymio field. 
               Thymio tag detection / Obstacles detection / Goal Detection 
        Input: image : Frame provided by the camera
        Output: field_map : image - vizualisation of the detections 
                thymio_tag : position of thymio tag center and 'nose' 
                goal : position of the goal 
                dilated_obstacles : detected and augmented obstacles
        """

        # Get Field 
        field = self.get_field(image)
        
        # Get Thymio Position 
        thymio_tag = self.thymio_detection(field)
        
        # Get Goal Position 
        goal = self.goal_detection(field)
        
        # Get Obstacles Positions 
        obstacles, dilated_obstacles = self.obstacles_detection(field)
        
        # Get Image for later display 
        field_map = self.get_field_map(field, thymio_tag, goal, obstacles, dilated_obstacles)
        
        return field_map, thymio_tag, goal, dilated_obstacles
    
            
    def get_field_map(self, field, thymio_tag, goal, obstacles, dilated_obstacles, blank=False) : 
        
        """
        Goal:  Build a visualization of the map after detection 
        Input: field = cropped and transformed camera frame to fit the thymio field 
               thymio_tag : position of the thymio tag (center and 'nose') 
               goal = position of the goal 
               obstacles = position of the dilated obstacles 
               blank = visualize detection on top of the cropped image or a blank screen
        Output: field_map = image - vizualisation of the detections 
        """
        
        # Empty map or superimpose 
        if blank :
            field_map = (255*np.ones(field.shape)).astype(np.uint8)
        else : 
            field_map = field.copy()
            
        # Add Thymio to map if detected 
        if thymio_tag : 
            cv2.circle(field_map, thymio_tag['pos'], 5, (0, 0, 255), -1)
            cv2.line(field_map, thymio_tag['pos'], thymio_tag['nose'], (255, 0, 255), 5)

        # Add Goal to map 
        cv2.circle(field_map, goal['pos'], 5, (255, 0, 0), -1)
        
        # Add obstacles to map 
        for (idx, corners_list) in obstacles.items() :   
            pts = np.array(corners_list).reshape((-1, 1, 2))
            cv2.polylines(field_map, [pts], True,  (0, 0, 255, 5))   
            for corners in corners_list : 
                cv2.circle(field_map, [corners[0], corners[1]], 5, (166, 235, 255), 3)
                
        # Add dilated obstacles to map 
        for (idx, corners_list) in dilated_obstacles.items() :   
            pts = np.array(corners_list).reshape((-1, 1, 2))
            cv2.polylines(field_map, [pts], True,  (255, 0, 0, 5))   
            for corners in corners_list : 
                cv2.circle(field_map, [corners[0], corners[1]], 5, (255, 173, 166), 3)

        return field_map

    
    def draw_thymio_tag(self, field_map, thymio_tag) : 
        
        """
        Goal:  Update the thymio tag position and orientation for live vizualisation 
        Input: field_map = initial map where obstacles and goal detection has already been performed 
        Output: field_map = updated map with thymio tag drawn 
        """
        
        if thymio_tag is not None  : 
            cv2.circle(field_map, thymio_tag['pos'], 5, (0, 0, 255), -1)
            cv2.line(field_map, thymio_tag['pos'], thymio_tag['nose'], (255, 0, 255), 5)
            
        return field_map 

    
    def obstacles_detection(self, field, thymio_clearance=150) : 
        
        """
        Goal:  Detected obstacles 
        Input: field = cropped and transformed camera frame to fit the thymio field 
               thymio_clearance = how much the obstacles should be increased considering the thymio size (mm) 
        Output: obstacles_dict = dictionnary of obstacles corners positions 
                dilated_obstacles_dict = dictionnary of dilated obstacles corners positions 
        """
        
        # Store Obstacle Shapes
        obstacles_dict = {}
        dilated_obstacles_dict = {}
        
        # Get obstacles 
        red_filter = self.red_filtering(field)
        
        # Bitwise Inverse
        obstacles = cv2.bitwise_not(red_filter[:, :, 0])
        
        # Expand Obstacles
        kernel = np.ones((thymio_clearance, thymio_clearance), np.uint8)
        dilated_obstacles = cv2.dilate(obstacles, kernel, iterations=1)
        
        # Find contours of filtered shapes
        contours, _  = cv2.findContours(obstacles, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        dilated_contours, _ = cv2.findContours(dilated_obstacles, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Compute corners from contours 
        for i, contour in enumerate(contours):
            #if i == 0: continue
            # Approximate Shape 
            approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
            obstacles_dict[i] = [(corners[0][0], corners[0][1]) for corners in approx]
         
        # Compute corners from contours
        for j, contour in enumerate(dilated_contours):
            #if j == 0: continue
            # Approximate Shape 
            approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
            dilated_obstacles_dict[j] = [(corners[0][0], corners[0][1]) for corners in approx]
            
        return obstacles_dict, dilated_obstacles_dict

    
    def thymio_detection(self, field) : 
        
        """
        Goal:  Detect the thymio tag  
        Input: field = cropped and transformed camera frame to fit the thymio field 
        Output: thymio tag = dictionnary with positions of thymio tag center and 'nose' 
                -- The thymio tag 'nose' is the middle position between its top right and top left corners --
        """
        
        
        # Thymio Tag Position 
        thymio_tag = {'pos' : None, 'nose': None}
        
        # Tag Detection 
        options  = apriltag.DetectorOptions(families="tag25h9")
        detector = apriltag.Detector(options)
        results  = detector.detect(cv2.cvtColor(field, cv2.COLOR_BGR2GRAY)) 
        
        # Results 
        if len(results)==0 : 
            print('Thymio Not Detected')
            return None
        else :   
            r = results[0]
            (ptA, ptB, ptC, ptD) = r.corners
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))
            ptA = (int(ptA[0]), int(ptA[1]))

            thymio_tag['pos']  = (int(r.center[0]), int(r.center[1]))
            thymio_tag['nose'] = (int((ptA[0]+ptB[0])/2), int((ptA[1]+ptB[1])/2))
            return thymio_tag
        
        
    def goal_detection(self, field) : 
        
        """
        Goal:  Detected obstacles 
        Input: field = cropped and transformed camera frame to fit the thymio field 
        Output: goal = dictionnary containing position of the goal, position is set to origin in case of detection failure
        """
        
        # Goal Position 
        goal = {}
        
        # Filtering of green values
        green_filter = self.green_filtering(field)
        contours, _  = cv2.findContours(green_filter, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Test detection 
        if len(contours) == 1  : 
            print('Goal Not Detected')
            goal['pos'] = (0, 0)
            return goal
        else :          
            # Get center of green shape
            M = cv2.moments(contours[1])
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"]) 
            goal['pos'] = (cX, cY)
        
            return goal
        
    
    
    def get_field(self, image) : 
        
        """
        Goal:  Crop the camera frame to fit the thymio field 
        Input:  image = camera frame 
        Output: field = cropped and transformed camera frame to fit the thymio field 
        """
        
        # Field Size 
        width  = self.field_size['width']
        height = self.field_size['height']
        
        # Image 
        rows,cols,ch = image.shape
        pts1 = np.float32([self.crop['top_left'], self.crop['top_right'], self.crop['bottom_left'], self.crop['bottom_right']])
        pts2 = np.float32([[0,0],[width, 0],[0, height],[width,height]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        field = cv2.warpPerspective(image, M, (width,height))
        
        return field 
    
    
    
    
    ## ----- Corners Detection ----- ## 
    
        
    def detect_corner_tags(self, image, display=False) : 
         
        """
        Goal:  Detect the april tags for each corners
        Input:  image = camera frame
                display = display the corner tags detection for debugging 
        Output: True/False -- returns True when all four corners are detected 
        """
    
        # Get Black Filtered Input 
        black_filter = self.black_filtering(image) 
        
        # Tag Detection 
        options  = apriltag.DetectorOptions(families="tag36h11")
        detector = apriltag.Detector(options)
        results  = detector.detect(black_filter)   
        
        # Visualization 
        if display : self.display_corner_tags_detection(image, results) 
        
        # Get Tag Corners  
        ww = ['right'  if a[0] > np.mean([r.center[0] for r in results]) else 'left' for a in [r.center for r in results]]
        hh = ['bottom' if a[1] > np.mean([r.center[1] for r in results]) else 'top' for a in [r.center for r in results]]
        positions = [h + '_' + w for h, w in zip(hh, ww)]
        tag_dict = dict(zip(positions, [r for r in results]))
        cropping = {}

        # Get Field Corners  
        for position, tag in tag_dict.items() : 
            hh = ['bottom' if y>np.mean([x[1] for x in tag.corners]) else 'top' for y in [x[1] for x in tag.corners]]
            ww = ['right' if y>np.mean([x[0] for x in tag.corners]) else 'left' for y in [x[0] for x in tag.corners]]
            positions = [h + '_' + w for h, w in zip(hh, ww)]
            tag_corners_dict = dict(zip(positions, tag.corners))
            cropping[position] = tag_corners_dict[position]
            
        if len(cropping)!=4 : 
            self.crop = None 
            return False
        else : 
            self.crop = cropping
            return True
            
        
        
    def display_corner_tags_detection(self, image, results) : 
        
        """
        Goal:  Vizualisation of the corners tags detection 
        Input:  image = camera frame
                results = list of april tags objects returned by the april tag librairy 
        Output: None 
        """
        
        
        # Visualization 
        viz = image.copy()

        for r in results :

            (ptA, ptB, ptC, ptD) = r.corners
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))
            ptA = (int(ptA[0]), int(ptA[1]))

            # draw the bounding box of the AprilTag detection
            cv2.line(viz, ptA, ptB, (0, 255, 0), 2)
            cv2.line(viz, ptB, ptC, (0, 255, 0), 2)
            cv2.line(viz, ptC, ptD, (0, 255, 0), 2)
            cv2.line(viz, ptD, ptA, (0, 255, 0), 2)
            
            # draw the center (x, y)-coordinates of the AprilTag
            (cX, cY) = (int(r.center[0]), int(r.center[1]))
            cv2.circle(viz, (cX, cY), 5, (0, 0, 255), -1)
            # draw the tag family on the image
            tagFamily = r.tag_family.decode("utf-8")
            cv2.putText(viz, tagFamily, (ptA[0], ptA[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        #  AprilTag detection
        plt.figure(figsize=[15, 8])
        plt.title('Corners - April Tag Detection')
        plt.imshow(viz)
        plt.show()
        
        
        
        
    ## ---- Filterings ---- ##     
        
    def black_filtering(self, image, display=True) : 
        
        """
        Goal:  Filter black values to increase april tag detection efficiency
        Input:  image = camera frame
                display = display the black filter for debugging  
        Output: black_filter = one-channel image matrix - thresholded filter on black values
        """
        
        # Filtering Step to get goal 
        black_filter = np.zeros(image.shape)
        red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2] 
        
        # Get Thresholds 
        red_th   = self.black_th['red']
        green_th = self.black_th['green']
        blue_th  = self.black_th['blue']

        # Color filtering
        color_filter = np.logical_and(red<red_th, green<green_th, blue<blue_th)
        black_filter[:, :, 0] = 255*(1-color_filter).astype(int)
        black_filter[:, :, 1] = 255*(1-color_filter).astype(int)
        black_filter[:, :, 2] = 255*(1-color_filter).astype(int)
        black_filter = black_filter.astype(np.uint8)

        # Display 
        if display : 
            plt.figure(figsize=[15, 4])
            plt.imshow(black_filter)
            plt.title('Filtering Black')
            plt.show()
            
        return black_filter[:, :, 0]
    

    def red_filtering(self, field, display=False) : 
        
        """
        Goal:  Filter red values as a first step in obstacle detection 
        Input:  image = camera frame
                display = display the red filter for debugging  
        Output: red_filter = three-channel image matrix - thresholded filter on red values
        """
        
        # Filtering Step to get goal 
        red_filter = np.zeros(field.shape)
        red, green, blue = field[:, :, 0], field[:, :, 1], field[:, :, 2] 
        
        # Get Thresholds 
        red_th   = self.red_th['red']
        green_th = self.red_th['green']
        blue_th  = self.red_th['blue']

        # Color filtering
        color_filter = np.logical_and(red>red_th, green<green_th, blue<blue_th)
        red_filter[:, :, 0] = 255*(1-color_filter).astype(int)
        red_filter[:, :, 1] = 255*(1-color_filter).astype(int)
        red_filter[:, :, 2] = 255*(1-color_filter).astype(int)
        red_filter = red_filter.astype(np.uint8)

        # Display 
        if display : 
            plt.imshow(red_filter)
            plt.title('Filtering Red')
            plt.show()
            
        return red_filter
    
    
    def green_filtering(self, field, display=False) : 
        
        """
        Goal:  Filter green values as a first step in goal detection 
        Input:  image = camera frame
                display = display the green filter for debugging  
        Output: green_filter = one-channel image matrix - thresholded filter on green values
        """
    
        # Filtering Step to get goal 
        green_filter = np.zeros(field.shape)
        red, green, blue = field[:, :, 0], field[:, :, 1], field[:, :, 2] 
        
        # Get Thresholds 
        red_th   = self.green_th['red']
        green_th = self.green_th['green']
        blue_th  = self.green_th['blue']

        color_filter = np.logical_and(red<red_th, green>green_th, blue<blue_th)
        green_filter[:, :, 0] = 255*(1-color_filter).astype(int)
        green_filter[:, :, 1] = 255*(1-color_filter).astype(int)
        green_filter[:, :, 2] = 255*(1-color_filter).astype(int)
        green_filter = green_filter.astype(np.uint8)

        # Additional Filtering Pass 
        kernel = np.ones((5,5),np.float32)/25
        green_filter = cv2.filter2D(green_filter,-1,kernel)
        _, green_filter = cv2.threshold(green_filter,127,255,cv2.THRESH_BINARY)
        
        # Display 
        if display : 
            plt.imshow(green_filter)
            plt.title('Filtering Green')
            plt.show()
            plt.imshow(field)
            plt.show()
            
        return green_filter[:, :, 0]
    
    
    
    
    ## ------------- NOT USED ------------- ##
    
    def get_occupancy_grid(self, field, resolution, display=True, thymio_clearance=180) : 
        
        """
        Goal:  Outputs an occupancy grid with a given resolution
        """
        
        #Resolution - nb of mm2 
        width = self.field_size['width']//resolution
        height = self.field_size['height']//resolution
        
        # Get obstacles 
        red_filter = self.red_filtering(field)
        
        # Expand Obstacles
        kernel = np.ones((thymio_clearance, thymio_clearance), np.uint8)
        dilated_obstacles = cv2.dilate((-1)*red_filter+255, kernel, iterations=1)
        
        # Resize input to "pixelated" size
        occupancy = cv2.resize(dilated_obstacles, (width, height), interpolation=cv2.INTER_LINEAR)
        
        if display : 
            plt.figure(figsize=[10, 10])
            plt.title('Occupancy Grid - Resolution {0}mm2'.format(resolution))
            plt.imshow(cv2.resize(occupancy, (width*resolution, height*resolution), interpolation=cv2.INTER_NEAREST))
            plt.show()
    
        return occupancy[:, :, 0]

        
        
        
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        