import math
import numpy as np
import time

               
class Kalman:
    
    def __init__(self,thymio): 
        
        """
        Goal:   Initialization of a Kalman class that is used to perform the Kalman filtering 
        Input:  thymio = thymio class with the data of our robot
        Output: kalman class created from the information of the Thymio
        """
        
        #Transition matrix state
        self.A = np.matrix([[1,0,0],[0,1,0],[0,0,1]],dtype= 'float')
        #Transition matrix input
        self.B = np.matrix([[1,0,0],[0,1,0],[0,0,1]],dtype= 'float')
        #State of the Thymio
        self.E = np.matrix([[thymio.x_cam],[thymio.y_cam],[thymio.theta_cam]],dtype= 'float')
        #Input of the wheels (Motor speeds)
        self.U = np.matrix([[0],[0]],dtype= 'float')
        #Variances of motors
        u1 = 19
        u2 = 19
        self.U_var_m = np.diag([u1,u2])
        #Variances of measurement (x,y,theta)
        r1 = 1
        r2 = 1
        r3 = 2*np.pi/360
        self.R = np.matrix([[r1,0,0],[0,r2,0],[0,0,r3]],dtype= 'float')
        self.P = np.matrix([[10,0,0],[0,10,0],[0,0,1]],dtype= 'float')
        #Measurement matrix 
        self.H = np.matrix([[1,0,0],[0,1,0],[0,0,1]],dtype= 'float')
        #Distance between the two wheels
        self.b = 80
        #Delta time 
        self.lastKalman = time.time_ns()/10e8
        # Constant used in the odometry formula to change speed from pwm to mm/s
        self.c = 32/100
    
    
    def kalman_prediction(self,thymio,L_speed,R_speed):
        
        """
        Goal:     Since at this point we do not have the measurement done by the webcam, it
                  predicts the state of the robot based on the odometry and it updates the 
                  variances of the system.
        Input:    thymio = thymio class with the data of our robot
                  L_speed = speed of the left wheel applied until now
                  R_speed = speed of the right wheel applied until now
        Output:   -
        """
        
        # Compute the time between the last update/prediction and this time
        deltaT = time.time_ns()/10e8 - self.lastKalman
        # Compute the transition matrix of the input corresponding to the odometry formula
        self.B = np.matrix([[0.5*math.cos(self.E[2])*self.c,0.5*math.cos(self.E[2])*self.c],
                            [0.5*math.sin(self.E[2])*self.c,0.5*math.sin(self.E[2])*self.c],
                            [1/self.b*self.c, -1/self.b*self.c]],dtype= 'float')
        
        # Compute the predicted state of the robot (AE+BU)
        self.E = np.dot(self.A, self.E) + np.dot(self.B, self.U)*deltaT
        
        # Compute the additionnal uncertainty due to the motors
        Q = np.dot(self.B, np.dot(self.U_var_m,self.B.T)) +np.eye(3)
        # Update the variance of the system
        self.P = np.dot(np.dot(self.A,self.P),np.transpose(self.A))+Q
         
        # Update the state with the predicted x, y and theta
        thymio.x_est = self.E[0].item()
        thymio.y_est = self.E[1].item()
        thymio.theta_est = self.E[2].item()
        
        # Keep the speeds of the robot in U to compute the next prediction
        self.U = np.matrix([[L_speed],[R_speed]],dtype= 'float')
        
        # Update the time of the last kalman done to find deltaT
        self.lastKalman = time.time_ns()/10e8

        
            
    def kalman_update(self,thymio):
        """
        Goal:   Since we have the measurement of the camera, it can update 
                the state of the robot and update the variances of the system.
        Input:  thymio = thymio class with the data of our robot
        
        Output: returns false to set the boolean that says that the last block 
                was not the vision block.
        """
        
        # Put the measurement of the webcam in the measured state matrix
        Z = np.matrix([[thymio.x_cam],[thymio.y_cam],[thymio.theta_cam]],dtype= 'float')
        
        # Update the state of the kalman with the state of the robot
        self.E[0] = thymio.x_est 
        self.E[1] = thymio.y_est 
        self.E[2] = thymio.theta_est 
        
        # Computation of the Kalman gain 
        K_den = np.linalg.inv(np.dot(self.H,np.dot(self.P,np.transpose(self.H))) + self.R)
        K_num = np.dot(self.P,np.transpose(self.H))
        K = np.dot(K_num,K_den)        
        
        #Correction of the state with the Kalman gain and the measurements
        self.E = self.E + np.dot(K,(Z-np.dot(self.H,self.E)))
        
        #Update of the variance of the system
        I = np.eye(3)
        self.P = np.dot((I-np.dot(K,self.H)),self.P)
            
        # Update the state with the corrected x, y and theta
        thymio.x_est = self.E[0].item()
        thymio.y_est = self.E[1].item()
        thymio.theta_est = self.E[2].item()
        
        # Update the time of the last kalman done to find deltaT
        self.lastKalman = time.time_ns()/10e8

        return False