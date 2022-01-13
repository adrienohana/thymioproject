#Main Loop no. of position updates for one vision step
MOTION_RATIO = 10
#counter for main control loop
count = 0


# Detection Parameters
black_th = {'red':105, 'green':105, 'blue':105}
red_th   = {'red':140, 'green':100, 'blue':130}
green_th = {'red':30, 'green':50, 'blue':52}
field_size = {'height': 210*5, 'width': 297*4}

# Obstacles and Deviation Gains 
devGain = 60
speed = 50

# Scale factors for sensors and constant factor
sensor_scale = 180

#weight matrices for local avoidance
w_l = [6, 4, -2, -6, -8, 2, 0]
w_r = [-6, -4, -2, 6, 8, 0, 2]

#front sensors + memory
x = [0,0,0,0,0,0,0]
#memory decay must be > 1
mem_decay = 80

#speed initializations
L_speed = 0
R_speed = 0
L_speed_goal = 0
R_speed_goal = 0
L_speed_obstacle = 0
R_speed_obstacle = 0

#boolean for Kalman update or prediction
boolKF = False