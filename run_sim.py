import os
import math
import mujoco
import signal
import mujoco_viewer
import numpy as np
import matplotlib.pyplot as plt

def soft_start(t,ts):
    """
    Soft startup function for actuators. Ramps from 0.0 to 1.0 during interval from t=0
    to t=ts.
    """
    rval = 0.0
    if t < ts:
        rval = t/ts
    else:
        rval = 1.0
    return rval


def sin_kinematics(t, amp, phase, period):
    """
    Sine wave kinematics. Returns potition and velocity for PD controller. 
    """
    pos = amp*math.sin(2.0*math.pi*(t + phase)/period)
    vel = (2.0*math.pi/period)*amp*math.cos(2.0*math.pi*(t + phase)/period)
    return pos, vel


done = False

def sigint_handler(signum, frame):
    """
    SIGINT handler. Sets done to True to quit simulation.
    """
    global done
    done = True

# ------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    period = 1.0

    upper_amp = 0.090
    lower_amp = 0.085

    upper_phase = 0.00*period
    lower_phase = 0.17*period 

    startup_frac = 1.5

    model = mujoco.MjModel.from_xml_path('wing_hinge.xml')
    data = mujoco.MjData(model)
    viewer = mujoco_viewer.MujocoViewer(model, data)
    signal.signal(signal.SIGINT, sigint_handler)

    t = []
    upper_slider_data = []
    lower_slider_data = []
    wing_joint_x_data = []
    wing_joint_z_data = []

    while not done:

        mujoco.mj_step(model, data)
        viewer.render()
        startup_val = soft_start(data.time, startup_frac*period)
        upper_pos, upper_vel = sin_kinematics(data.time, upper_amp, upper_phase, period)
        data.actuator('upper_position_servo').ctrl = upper_pos*startup_val
        data.actuator('upper_velocity_servo').ctrl = upper_vel*startup_val

        lower_pos, lower_vel = sin_kinematics(data.time, lower_amp, lower_phase, period)
        data.actuator('lower_position_servo').ctrl = lower_pos*startup_val
        data.actuator('lower_velocity_servo').ctrl = lower_vel*startup_val

        t.append(data.time)
        lower_slider_data.append(data.joint('actuator_lower_slider').qpos[0])
        upper_slider_data.append(data.joint('actuator_upper_slider').qpos[0])
        wing_joint_x_data.append(data.joint('wing_joint_x').qpos[0])
        wing_joint_z_data.append(data.joint('wing_joint_z').qpos[0])

    viewer.close()

    lower_slider_data = np.array(lower_slider_data)
    upper_slider_data = np.array(upper_slider_data)
    wing_joint_x_data = np.rad2deg(np.array(wing_joint_x_data))
    wing_joint_z_data = np.rad2deg(np.array(wing_joint_z_data))

    fg, ax = plt.subplots(2,1,sharex=True)
    ax[0].plot(t, lower_slider_data)
    ax[0].plot(t, upper_slider_data)
    ax[0].set_ylabel('slider len')
    ax[0].grid(True)

    ax[1].plot(t, wing_joint_x_data)
    ax[1].plot(t, wing_joint_z_data)
    ax[1].set_ylabel('wing ang')
    ax[1].set_xlabel('t (sec)')
    ax[1].grid(True)

    plt.show()

