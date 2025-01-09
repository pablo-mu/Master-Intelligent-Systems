from GUI import GUI
from HAL import HAL
import cv2
import math
import numpy as np


# Enter sequential code!
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
VICTIMS_X = 30
VICTIMS_Y = -40.3
ship_world = (0,0)
survivors_world = (VICTIMS_X, VICTIMS_Y)

#search area definition
spiral_separation = 1.5 #we want that the radius expand 4 metres in each rotation
angle_step = -math.pi / 2 # 90 degrees per step for a rapid spiral expansion
num_steps = 100
fly_height = 1
search_path = []
survivor_cords = []

# Generate the search path. Tengo dudas de si esto se utilizará para el comando de mover o para simplemente decir donde tiene que ir.
for step in range(num_steps):
    radius = step * spiral_separation
    angle = step * angle_step
    x = radius * math.cos(angle)+ survivors_world[0] #coordenates are related to the initial position
    y = radius * math.sin(angle) + survivors_world[1]
    search_path.append((x, y))


def go_survivors_zone():
    x_pos = HAL.get_position()[0]
    y_pos = HAL.get_position()[1]
    #First here is searching the people
    while not ((VICTIMS_X -1 < x_pos) and (x_pos < VICTIMS_X +1 ) and (VICTIMS_Y -1< y_pos) and (y_pos < VICTIMS_Y +1 )):
        GUI.showImage(HAL.get_frontal_image())
        GUI.showLeftImage(HAL.get_ventral_image())
        x_pos = HAL.get_position()[0]
        y_pos = HAL.get_position()[1]
        HAL.set_cmd_pos(VICTIMS_X, VICTIMS_Y, fly_height, 0)
    return print("We are in the zone of the survivors")

def go_land_zone():
    x_pos = HAL.get_position()[0]
    y_pos = HAL.get_position()[1]
    #First here is searching the people
    while not np.linalg.norm((x_pos,y_pos))<0.5:
        GUI.showImage(HAL.get_frontal_image())
        GUI.showLeftImage(HAL.get_ventral_image())
        x_pos = HAL.get_position()[0]
        y_pos = HAL.get_position()[1]
        HAL.set_cmd_pos(0, 0, fly_height, 0)
    return print("We are ready to land off")

def land():
    HAL.land()
    while HAL.get_landed_state()!=1:
        GUI.showImage(HAL.get_frontal_image())
        GUI.showLeftImage(HAL.get_ventral_image())
        print('The drone is landing')
    return True  
    
#now we are going to tell the dron, how to move. #first we we'll define a function that indicates if we are in the desired
#position
def drone_in_pos(current_goal, fly_height):
    x_pos = HAL.get_position()[0]
    y_pos = HAL.get_position()[1]   	 
    if ((current_goal[0] -0.75 < x_pos) and (x_pos < current_goal[0] +0.75 ) and (current_goal[1]-0.75 < y_pos) and (y_pos < current_goal[1] + 0.75 ) and (abs(HAL.get_yaw())< fly_height)):
        return True
    else:
        return False
   	 
def rotate_image(img, angle):
    height, width = img.shape[:2]
    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
    # Apply the rotation to the image
    rotated_image = cv2.warpAffine(img, rotation_matrix, (width, height))
    return rotated_image
    

      
def location_survivor(face_cords):
  #Here, we check if is a repeated face (we check if we have a very close coords)
    count = []
    if len(survivor_cords) >= 1 :
        for cordenates in survivor_cords:
            if abs(face_cords[0]-cordenates[0])+abs(face_cords[1] - cordenates[1]) < 6:
                count.append(True)
            else:
                count.append(False)
        if sum(count)== 0:
            survivor_cords.append((face_cords[0],face_cords[1]))
            return print(f"There is a survivor in ({face_cords[0]}, {face_cords[1]})")
        else:
            return print(f"We already know the coordinates of this survivor")
    else:
        survivor_cords.append((face_cords[0], face_cords[1]))
        return print(f"There is a survivor in ({face_cords[0]}, {face_cords[1]})")
    
      

def check_survivors():
    faces = []
    img = HAL.get_ventral_image()
    for angle in range(0, 360, 10):
        rotated_img = rotate_image(img, angle)
        gray = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(face) >=1:
            faces.append(face)
          #aqui habría que almacenar la posición del superviviente.
    if len(faces)>=1:
        x_cord = HAL.get_position()[0]
        y_cord = HAL.get_position()[1]
        return location_survivor((x_cord, y_cord))
    else:
        return None

    
#Takeoff at the current location, to the given height
HAL.takeoff(fly_height)



error = HAL.get_yaw()
landed = False
while not landed:
    go_survivors_zone()
    search_path = search_path[1:]
    while len(survivor_cords) < 6:
        check_survivors()
        current_goal = search_path[0]
        #print(f'current goal: {current_goal},my position:{(HAL.get_position()[0],HAL.get_position()[1], HAL.get_yaw())}')
        #print(f'current goal: {current_goal}velocity:{(HAL.get_velocity()[0],HAL.get_velocity()[1], HAL.get_yaw())}')
        if drone_in_pos(current_goal, fly_height):
            print('Changing the path')
            search_path = search_path[1:]
            HAL.set_cmd_vel(0,0,0,-error)
        else:
            error_vel_x = current_goal[0]-HAL.get_position()[0]
            error_vel_y = current_goal[1]-HAL.get_position()[1]
            error_vec = (error_vel_x, error_vel_y)
            HAL.set_cmd_vel(error_vel_x/np.linalg.norm(error_vec),error_vel_y/np.linalg.norm(error_vec),-1*(HAL.get_position()[2]-3), -error)
            error = HAL.get_yaw()
        GUI.showImage(HAL.get_frontal_image())
        GUI.showLeftImage(HAL.get_ventral_image())
    print("We have all the survivors")
    go_land_zone()
    landed = land()
while True:
    print(f"We've already finished!, the survivors cordinates are {survivor_cords}")
