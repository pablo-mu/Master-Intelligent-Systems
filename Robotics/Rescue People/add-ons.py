#Basic PD for stabilization of the drone
# Define stabilization parameters
Kp = 0.5  # Proportional gain
Kd = 0.1  # Derivative gain

# Initialize variables
previous_roll = 0
previous_pitch = 0
previous_yaw = 0

def stabilize_drone():

    # Read drone's attitude data
    roll, pitch, yaw = HAL.get_attitude()

    # Calculate error terms
    roll_error = roll - desired_roll
    pitch_error = pitch - desired_pitch
    yaw_error = yaw - desired_yaw

    # Calculate control signals using PID controller
    roll_control = Kp * roll_error + Kd * (roll - previous_roll)
    pitch_control = Kp * pitch_error + Kd * (pitch - previous_pitch)
    yaw_control = Kp * yaw_error + Kd * (yaw - previous_yaw)

    # Apply control signals to drone's motors
    HAL.set_motor_commands(roll_control, pitch_control, yaw_control)

    # Update previous attitude values
    previous_roll = roll
    previous_pitch = pitch
    previous_yaw = yaw

