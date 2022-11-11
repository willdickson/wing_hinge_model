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


def sin_motion(t, amplitude, phase, offset, period):
    start_value  = soft_start(data.time, period)
    pos = amplitude*math.sin(2.0*math.pi*(t + phase)/period) + offset
    vel = (2.0*math.pi/period)*amplitude*math.cos(2.0*math.pi*(t + phase)/period)
    pos *= start_value
    vel *= start_value
    return pos, vel


# ------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    model = mujoco.MjModel.from_xml_path('model.xml')
    data = mujoco.MjData(model)

    viewer = mujoco_viewer.MujocoViewer(model, data)
    viewer.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    viewer.render_mode = 'window'
    #viewer.render_mode = 'offscreen'

    signal.signal(signal.SIGINT, sigint_handler)
    
    viewer.cam.distance = 500 
    viewer.cam.azimuth =  90 
    viewer.cam.elevation = -4
    #viewer.cam.lookat = [20.0, -15.0, 230.0]
    #viewer.cam.lookat = [100.0, -100.0, 230.0]
    viewer.cam.lookat = [0.0, 0.0, 230.0]
    

    period = 2.0

    scu_to_sla_scale = 1.0*math.acos(1.0/30.0)
    scu_amplitude = 5.0
    scu_phase = 0.5*period
    scu_offset = -2.0 
    sla_amplitude = 1.1*np.deg2rad(scu_to_sla_scale*scu_amplitude)
    sla_phase = 0.20*period
    sla_offset = -np.deg2rad(scu_to_sla_scale*scu_offset)

    ax4_amplitude = -1.0
    ax4_phase = 0.0
    ax4_offset = 0.0

    ax4_kp_gain = 80.0
    ax4_kv_gain = 30.0

    t_list = []
    ax4_pos_list = []
    ax4_setp_pos_list = []

    phi_list = []
    alpha_list = []
    theta_list = []
    

    while not done:

        mujoco.mj_step(model, data)
        
        sla_setp_pos, sla_setp_vel = sin_motion(data.time, sla_amplitude, sla_phase, sla_offset, period)
        data.actuator('sla_001_position').ctrl = sla_setp_pos
        data.actuator('sla_001_velocity').ctrl = sla_setp_vel

        scu_setp_pos, scu_setp_vel = sin_motion(data.time, scu_amplitude, scu_phase, scu_offset, period)
        data.actuator('scu_001_position').ctrl = scu_setp_pos
        data.actuator('scu_001_velocity').ctrl = scu_setp_vel

        if 0:
            ax4_setp_pos, ax4_setp_vel = sin_motion(data.time, ax4_amplitude, ax4_phase, ax4_offset, period)
            data.actuator('ax4_001_position').ctrl = ax4_setp_pos
            data.actuator('ax4_001_velocity').ctrl = ax4_setp_vel

        if 1:
            theta = data.joint('ax1_and_vein_001_joint_theta').qpos[0]
            dtheta_dt = data.joint('ax1_and_vein_001_joint_theta').qvel[0]
            ax4_pos = data.joint('ax4_001_joint').qpos[0]
            ax4_setp_pos = ax4_kp_gain*theta
            ax4_setp_vel = ax4_kv_gain*dtheta_dt
            data.actuator('ax4_001_position').ctrl = ax4_setp_pos
            data.actuator('ax4_001_velocity').ctrl = ax4_setp_vel

        try:
            if viewer.render_mode == 'window':
                frame = None
                viewer.render()
            else:
                frame = viewer.read_pixels()
        except:
            done = True

        if 0:
            sla_pos = data.joint('sla_001_joint').qpos[0]
            sla_err = sla_setp_pos - sla_pos
            print(f'{np.rad2deg(sla_setp_pos):1.4f}, {np.rad2deg(sla_pos):1.4f}, {np.rad2deg(sla_err):1.4f}')

        if 0:
            scu_pos = data.joint('scu_001_joint').qpos[0]
            scu_err = scu_setp_pos - scu_pos
            print(f'{scu_setp_pos:1.4f}, {scu_pos:1.4f}, {scu_err:1.4f}')
        if 0:
            ax4_pos = data.joint('ax4_001_joint').qpos[0]
            ax4_err = ax4_setp_pos - ax4_pos
            print(f'{ax4_setp_pos:1.4f}, {ax4_pos:1.4f}, {ax4_err:1.4f}')
        if 1:

            print(data.time)
            print(f'distance:  {viewer.cam.distance}')
            print(f'azimuth:   {viewer.cam.azimuth}')
            print(f'elevation: {viewer.cam.elevation}')
            print(f'lookat:   {viewer.cam.lookat}')
            print()


        t_list.append(data.time)
        ax4_pos_list.append(ax4_pos)
        ax4_setp_pos_list.append(ax4_setp_pos)

        phi_list.append(data.joint('ax1_and_vein_001_joint_phi').qpos[0])
        alpha_list.append(data.joint('ax1_and_vein_001_joint_alpha').qpos[0])
        theta_list.append(data.joint('ax1_and_vein_001_joint_theta').qpos[0])

    viewer.close()

t = np.array(t_list)
ax4_pos = np.array(ax4_pos_list)
ax4_setp_pos = np.array(ax4_setp_pos_list)

phi = np.rad2deg(np.array(phi_list))
phi = phi - phi.mean()
alpha = np.rad2deg(np.array(alpha_list))
theta = np.rad2deg(np.array(theta_list))

if 1:
    fig, ax = plt.subplots(1,1)
    phi_line, = ax.plot(t,phi,'b')
    alpha_line, = ax.plot(t,alpha,'g')
    theta_line, = ax.plot(t,theta,'r')
    ax.grid(True)
    ax.set_xlabel('t')
    ax.set_ylabel('angle')
    ax.legend((phi_line, alpha_line, theta_line), ('phi', 'alpha', 'theta'), loc='upper right')
if 1:
    fig, ax = plt.subplots(1,1)
    ax.plot(t,ax4_setp_pos, 'r')
    ax.plot(t,ax4_pos,'b')
    ax.grid(True)
    ax.set_xlabel('t')
    ax.set_ylabel('ax4_pos')
plt.show()


