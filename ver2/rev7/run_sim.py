import math
import pathlib
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
    
    #viewer.cam.distance = 186.0
    viewer.cam.distance = 120.0
    viewer.cam.azimuth =  144.0 
    viewer.cam.elevation = -27 
    #viewer.cam.lookat = np.array([20.0, -15.0, 230.0])
    #viewer.cam.lookat = np.array([100.0, -100.0, 230.0])
    #viewer.cam.lookat = np.array([100.0, 0.0, 230.0])
    viewer.cam.lookat = np.array([0.0, 0.0, 200.0])

    dt = model.opt.timestep

    period = 2.0
    steps_per_period = int(period/dt)
    print(f'steps_per_period {steps_per_period}')

    output_dir = pathlib.Path('frames')
    output_dir.mkdir(exist_ok=True)
    save_start = 2*steps_per_period
    save_stop = 3*steps_per_period
    save_step = 20
    save_list = list(range(save_start, save_stop, save_step))

    scu_to_sla_scale = 1.0*math.acos(1.0/30.0)
    scu_amplitude = 5.0
    scu_phase = 0.5*period
    #scu_offset = -2.0 
    scu_offset = -1.8 
    #sla_amplitude = 1.1*np.deg2rad(scu_to_sla_scale*scu_amplitude)
    sla_amplitude = 1.0*np.deg2rad(scu_to_sla_scale*scu_amplitude)
    sla_phase = 0.2*period
    sla_offset = -np.deg2rad(scu_to_sla_scale*scu_offset)

    #ax3_kp_gain = 80.0
    #ax3_kv_gain = 10.0
    ax3_kp_gain = 80.0
    ax3_kv_gain = 30.0

    t_list = []
    ax3_pos_list = []
    ax3_setp_pos_list = []

    phi_list = []
    alpha_list = []
    theta_list = []

    frame_count = 0

    while not done:

        mujoco.mj_step(model, data)
        nstep = int(data.time/dt)
        print(f'nstep {nstep}')
        
        sla_setp_pos, sla_setp_vel = sin_motion(data.time, sla_amplitude, sla_phase, sla_offset, period)
        data.actuator('sla_001_position').ctrl = sla_setp_pos
        data.actuator('sla_001_velocity').ctrl = sla_setp_vel

        scu_setp_pos, scu_setp_vel = sin_motion(data.time, scu_amplitude, scu_phase, scu_offset, period)
        data.actuator('scu_001_position').ctrl = scu_setp_pos
        data.actuator('scu_001_velocity').ctrl = scu_setp_vel

        theta = data.joint('ax1_and_vein_001_joint_theta').qpos[0]
        dtheta_dt = data.joint('ax1_and_vein_001_joint_theta').qvel[0]
        ax3_pos = data.joint('ax3_001_joint').qpos[0]
        ax3_setp_pos = ax3_kp_gain*theta
        ax3_setp_vel = ax3_kv_gain*dtheta_dt
        data.actuator('ax3_001_position').ctrl = ax3_setp_pos
        data.actuator('ax3_001_velocity').ctrl = ax3_setp_vel

        try:
            if viewer.render_mode == 'window':
                frame = None
                viewer.render()
            else:
                frame = viewer.read_pixels()
        except:
            done = True

        if frame is not None and nstep in save_list:
            filepath = pathlib.Path(output_dir, f'frame_{frame_count:06d}.png')
            print(f'saving: {nstep}, frame_count: {frame_count}, {filepath}')
            plt.imsave(filepath, frame)
            #np.save(filepath, frame)
            frame_count += 1
            if nstep == save_list[-1]:
                done = True

        if viewer.render_mode == 'window':
            print(f'time: {data.time}, nstep: {nstep}')
            print(f'distance:  {viewer.cam.distance}')
            print(f'azimuth:   {viewer.cam.azimuth}')
            print(f'elevation: {viewer.cam.elevation}')
            print(f'lookat:   {viewer.cam.lookat}')
            print()

        t_list.append(data.time)
        ax3_pos_list.append(ax3_pos)
        ax3_setp_pos_list.append(ax3_setp_pos)

        phi_list.append(data.joint('ax1_and_vein_001_joint_phi').qpos[0])
        alpha_list.append(data.joint('ax1_and_vein_001_joint_alpha').qpos[0])
        theta_list.append(data.joint('ax1_and_vein_001_joint_theta').qpos[0])

    viewer.close()

t = np.array(t_list)
ax3_pos = np.array(ax3_pos_list)
ax3_setp_pos = np.array(ax3_setp_pos_list)

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
    ax.plot(t,ax3_setp_pos, 'r')
    ax.plot(t,ax3_pos,'b')
    ax.grid(True)
    ax.set_xlabel('t')
    ax.set_ylabel('ax3_pos')

plt.show()


