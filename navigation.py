import pyvisgraph as vg
import cv2, math
import numpy as np

UNKNOWN = -1 


class Robot : 
    
    def __init__(self) : 
        """
        Goal:   Initialization of a Robot class containing state and global navigation variables
        """
        
        #Robot state (Theta relative to horizontal axis):
        #Latest camera position
        self.x_cam = None
        self.y_cam = None
        self.theta_cam = None
        #Latest position estimate from filtering
        self.x_est = None
        self.y_est = None
        self.theta_est = None
        
        # Global navigation waypoints 
        self.path = None
        self.next_goal = None
        self.finished=False
        
        #Robot constants
        #inter-wheel distance
        self.b = 95
        #distance from thymio tag to center of thymio
        self.tag2center=30
        
        
    def update_position(self, thymio_tag) :
        """
        Goal :  Update Thymio Position from detected thymio_tag position. Done only at initialization.
        Input : Thymio_tag of the form {'pos' : (x,y), 'nose': (x,y)}
        """
        #if the camera has found the thymio tag : convert and store state variables
        if thymio_tag is not None:
            
            theta = math.atan2(thymio_tag['nose'][1]-thymio_tag['pos'][1],thymio_tag['nose'][0]-thymio_tag['pos'][0])
            
            self.theta_cam = theta
            self.x_cam = thymio_tag['pos'][0] + self.tag2center*math.cos(theta)
            self.y_cam = thymio_tag['pos'][1] + self.tag2center*math.sin(theta)
            
            #initialize estimates with camera data
            self.x_est = self.x_cam
            self.y_est = self.y_cam
            self.theta_est = self.theta_cam
            
            
    def update_cam_pos(self, thymio_tag) :
        """
        Goal:   Update Thymio Position from detected thymio tag position. Done only after initialization.
        Input : Thymio_tag of the form {'pos' : (x,y), 'nose': (x,y)}
        """ 
        #convert and store state variables
        theta = math.atan2(thymio_tag['nose'][1]-thymio_tag['pos'][1],thymio_tag['nose'][0]-thymio_tag['pos'][0])

        self.theta_cam = theta
        self.x_cam = thymio_tag['pos'][0] + self.tag2center*math.cos(theta)
        self.y_cam = thymio_tag['pos'][1] + self.tag2center*math.sin(theta)
        
    
    def compute_deviation(self) : 
        '''
        Goal :   Compute angle difference between thymio and goal to use it in naviguation
        Output : Deviation angle (rad)
        '''
        #get next waypoint
        goal_pos = self.path[self.next_goal]
        #calculate angle between robot and goal
        omega = math.atan2(goal_pos.y-self.y_est, goal_pos.x-self.x_est)       
        
        #calculate difference with robot orientation and adjust angle to be in the interval [-pi,pi]
        if omega-self.theta_est<-np.pi :
            dev=-(omega-self.theta_est+np.pi)
        elif omega-self.theta_est>np.pi : 
            dev=-(omega-self.theta_est-np.pi)
        else :
            dev=omega-self.theta_est
        return dev
        
    def update_next_goal(self, tolerance=50, verbose=True) : 
        """
        Goal : Check distance with current goal and updates it if it comes sufficiently close
        Output : Return True if Goal Updated and False if not 
        """         
        #get next waypoint and calculate its distance to the robot
        goal_pos = self.path[self.next_goal]
        distance2goal = math.sqrt((self.x_est-goal_pos.x)**2+(self.y_est-goal_pos.y)**2)
        
        #update the next waypoint if the robot is close enough to the next waypoint
        if distance2goal < tolerance : 
            if verbose : print('Reached Goal {0}'.format(self.next_goal))
            if self.next_goal == len(self.path)-1 : 
                self.finished=True
                if verbose : print('Your journey is over little thymio')
            else :
                self.next_goal =  self.next_goal+1 
                if verbose : print('Now directing towards {0}'.format(self.next_goal))
            return True 
        else :
            return False 
        
        
    def path_planning(self, field_map, final_goal, obstacles) : 
        """
        Goal : Calculate the shortest path between the robot and the goal
        Input : field_map 2D array with shape of the size of the map in mm, final_goal['pos'] = (x,y), obstacles dictionary 
        Output : field_map with added path
        """                                   
        # Definition 
        start = vg.Point(self.x_est, self.y_est)
        objective  = vg.Point(final_goal['pos'][0], final_goal['pos'][1])

        # Calculate shortest path 
        polys = [[vg.Point(corners[0], corners[1]) for corners in obstacle] for (i, obstacle) in obstacles.items()]
        g = vg.VisGraph()
        g.build(polys)
        shortest = g.shortest_path(start, objective)
        
        # Define Path and Next Goal in Graph  
        self.path = shortest
        self.next_goal = 1
        
        # Update Field Map 
        field_map = self.draw_path(field_map)
        
        return field_map
    
    
    def draw_path(self, field_map) :
        """
        Goal : Draw the path on the field map visualization
        Input : field_map 2D array with shape of the size of the map in mm
        """   
        
        # Update Field Map with lines drawn between each waypoint
        path_lines = [[(int(self.path[i].x), int(self.path[i].y)), (int(self.path[i+1].x), int(self.path[i+1].y))] 
                      for i in range(len(self.path)-1)]
        for line in path_lines : 
            cv2.line(field_map, line[0], line[1], (255, 227, 166), 4)
            
            
    def draw_thymio(self, field_map) : 
        """
        Goal : Draw the robot on the field map visualization
        Input : field_map 2D array with shape of the size of the map in mm
        Output : field_map 2D array with shape of the size of the map in mm
        """     
        
        # Update Field Map with a circle drawn at the position of the robot
        if self.x_est != UNKNOWN and self.x_est is not None : 
            cv2.circle(field_map, [int(self.x_est), int(self.y_est)], 15, (249, 140, 170), -1)
            
        return field_map 