import os
import math
import mujoco
import signal
import mujoco_viewer
import numpy as np
import matplotlib.pyplot as plt


done = False

def sigint_handler(signum, frame):
    """
    SIGINT handler. Sets done to True to quit simulation.
    """
    global done
    done = True


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


def sin_motion(t, amplitude, phase, period):
    pos = amplitude*math.sin(2.0*math.pi*(t + phase)/period)
    vel = (2.0*math.pi/period)*amplitude*math.cos(2.0*math.pi*(t + phase)/period)
    return pos, vel


# ------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    model = mujoco.MjModel.from_xml_path('model.xml')
    data = mujoco.MjData(model)

    viewer = mujoco_viewer.MujocoViewer(model, data)
    signal.signal(signal.SIGINT, sigint_handler)
    
    viewer.cam.distance = 100
    viewer.cam.azimuth = 140 
    viewer.cam.lookat = [20.0, -15.0, 230.0]
    

    period = 1.0

    if 1:
        scu_to_sla_scale = math.acos(1.0/30.0)
        scu_amplitude = 8.0
        scu_phase = 0.5*period
        scu_offset = -1.8
        sla_amplitude = 1.23*np.deg2rad(scu_to_sla_scale*scu_amplitude)
        sla_phase = 0.15*period
        sla_offset = -np.deg2rad(scu_to_sla_scale*scu_offset)
    if 0:
        scu_to_sla_scale = 1.0*math.acos(1.0/30.0)
        scu_amplitude = 5.4
        scu_phase = 0.5*period
        scu_offset = -1.2 
        sla_amplitude = 1.5*np.deg2rad(scu_to_sla_scale*scu_amplitude)
        sla_phase = 0.15*period
        sla_offset = -np.deg2rad(scu_to_sla_scale*scu_offset)

    t_list = []
    phi_list = []
    alpha_list = []
    theta_list = []


    while not done:

        mujoco.mj_step(model, data)
        start_value  = soft_start(data.time, period)
        
        if data.time < 100*period:
            sla_pos, sla_vel = sin_motion(data.time, sla_amplitude, sla_phase, period)
            data.actuator('sla_001_position').ctrl = start_value*(sla_pos + sla_offset)
            data.actuator('sla_001_velocity').ctrl = start_value*sla_vel

            scu_pos, scu_vel = sin_motion(data.time, scu_amplitude, scu_phase, period)
            data.actuator('scu_001_position').ctrl = start_value*(scu_pos + scu_offset)
            data.actuator('scu_001_velocity').ctrl = start_value*scu_vel

        try:
            viewer.render()
        except:
            done = True

        sla_error_rad = sla_pos - data.joint('sla_001_joint').qpos[0]
        sla_error_deg = np.rad2deg(sla_error_rad)
        scu_error = scu_pos - data.joint('scu_001_joint').qpos[0]
        print(sla_error_deg, scu_error)

        t_list.append(data.time)

        phi_list.append(data.joint('ax1_and_vein_001_joint_phi').qpos[0])
        alpha_list.append(data.joint('ax1_and_vein_001_joint_alpha').qpos[0])
        theta_list.append(data.joint('ax1_and_vein_001_joint_theta').qpos[0])


    viewer.close()

t = np.array(t_list)
phi = np.rad2deg(np.array(phi_list))
phi = phi - phi.mean()
alpha = np.rad2deg(np.array(alpha_list))
theta = np.rad2deg(np.array(theta_list))

fig, ax = plt.subplots(1,1)
ax.plot(t,phi,'b')
ax.plot(t,alpha,'g')
ax.plot(t,theta,'r')
ax.grid(True)
ax.set_xlabel('t')
ax.set_ylabel('angle')
plt.show()


