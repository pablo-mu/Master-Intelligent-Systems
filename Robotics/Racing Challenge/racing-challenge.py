from GUI import GUI
from HAL import HAL
import cv2

# PID Controller Constants

prev_error_angular = 0.0
prev_error_accel = 0.0

integral_angular = 0.0
integral_accel = 0.0


accel = 0
curve_angle = 0

# PID for angular speed
def calculate_pid_angular(error):
    global prev_error_angular, integral_angular
    Kp = 0.2  # Proportional constant
    Ki = 0.0001 # Integral constant 
    Kd = -0.00002  # Derivative constant 
    proportional = error
    integral_angular += error
    derivative = error - prev_error_angular
    prev_error = error

    return Kp * proportional + Ki * integral_angular + Kd * derivative

# PID for acceleration
def calculate_pid_accel(error):
    global prev_error_accel, integral_accel
    Kp = 0.5  # Proportional constant
    Ki = 0  # Integral constant 
    Kd = 0  # Derivative constant 
    proportional = error
    integral_accel += error
    derivative = error - prev_error_accel
    prev_error = error

    return Kp * proportional + Ki * integral_accel + Kd * derivative


#Initializing values   
i = 0
cX, cY = 0,0


        
while True:

    # Image detection and processing
    img = HAL.getImage()
    

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv,
                      (0, 125, 125),
                      (30, 255, 255))
                         
    contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_contour)

    # Centroid based on the figure detected by contours
        M = cv2.moments(contours[0])
        if M["m00"] != 0:
            cX = M["m10"] / M["m00"]
            cY = M["m01"] / M["m00"]
        else:
            cX, cY = 0, 0

        # Error from centroid to the center of the car/screen
        error = 320 - cX

	# Initial speed=maximum speed (7)
        if i == 0:
            speed = 7 
            steering_angle = 0
	
	# Non-initial states
	# When the car is not centered, we get the angle of the curve
        if (cX >0):
            ellipse = cv2.fitEllipse(max_contour)
            curve_angle = ellipse[2]

	    # When it is not the initial state            
            if i > 0:

		# Define the angular speed (steering_angle) with the PID control function
                control_output_angular = calculate_pid_angular(error)
                steering_angle = 0.01*control_output_angular 

		# When the error is very big, we limit speed 0.8-2.5 (depending on the acceleration controlled by PID)
                if  abs(error) > 230:
                    error_accel = abs(error)
                    accel = 0.002*calculate_pid_accel(error_accel)
                    speed = min(max(0.8,speed-accel),2.5) #min limits in case the speed considering the acceleration exceeds the limits for this error

		# When the error is big, speed increases a bit 2.5-3.7 (depending on the acceleration controlled by PID)
                if abs(curve_angle) > 10 and abs(error) < 230:
                    error_accel = abs(error)
                    accel = 0.002*calculate_pid_accel(error_accel)
                    speed = min(max(2.5,speed+accel),3.7) #min limits in case the speed considering the acceleration exceeds the limits for this error
                   
		# When the error is smaller, speed can increase until maximum speed 7 (depending on the acceleration controlled by PID)
                if abs(curve_angle)<= 10 and abs(steering_angle)<0.05:
                    
                    error_accel = abs(error)
                    accel = 0.01*calculate_pid_accel(error_accel)
                    speed = min(max(3.7,speed+accel),7) #min limits in case the speed considering the acceleration exceeds the limits for this error

	    # Set the speeds previously calculated
            HAL.setW(steering_angle) 
            HAL.setV(speed)

        #Count the iteration of the loop and visualize values to control parameters     
        i = i+1  
        print('%d angle: %.2f speed: %.2f error: %.2f curve: %.2f accel: %.2f ' % ( i, steering_angle, speed, error, curve_angle, accel))

    else:
        # If no contours are found, stop the car
        HAL.setV(0)
        HAL.setW(0)     
        
    #Show the image    
    GUI.showImage(red_mask)
