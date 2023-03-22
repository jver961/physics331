import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

colors = ['red', 'blue', 'green']

# =============================================================================
# Input: tuple of arrays (represents position and velocity vectors)
# Output: single concatenated array (represents state vector)
def State(*args):
    out = args[0]
    for v in args[1:]:
        out = np.concatenate((out, v))
    return out

# =============================================================================
# Input: current time t, current state vector state, array of masses
# Output: derivative vector at (t, state)
def derivative(t, state, masses):
    
    particle_num = state.shape[0]//6
    
    # Array of distances
    dist = np.zeros((particle_num, particle_num))
    for i in range(particle_num):
        for j in range(particle_num):
            dist[i, j] = np.linalg.norm(state[(6*i):(6*i+3)] - state[(6*j):(6*j+3)])
    
    output = np.zeros(len(state))
    for i in range(particle_num):
        
        output[(6*i):(6*i+3)] = state[(6*i+3):(6*i+6)]
        
        Dvi = np.zeros(3)
        for j in range(particle_num):
            if j != i:
                Dvi += masses[j]/(dist[i,j]**3)*(state[(6*j):(6*j+3)] - state[(6*i):(6*i+3)])
        
        output[(6*i+3):(6*i+6)] = Dvi
    
    return output

# =============================================================================
# The following three functions calculate the COM, energy, and angular momentum
# from the system parameters.

# Input: vector parameter for a system (format [particle, time, component])
# Output: mass-weighted average at each time (format [time, component])
def COM(array, masses):
    out = np.cumsum(np.transpose(np.transpose(array)*masses),axis=0)[-1]
    return out/sum(masses)

# Input: position and velocity (format [particle, time, component]) and masses
# Output: kinetic energy at each time (format [time])
def energy(xarray, varray, masses):
    out = np.zeros(xarray.shape[1])
    for i in range(xarray.shape[0]):
        out += 0.5*masses[i]*np.linalg.norm(varray[i], axis=1)**2
        for j in range(i):
            out -= masses[i]*masses[j]/np.linalg.norm(xarray[i]-xarray[j], axis=1)
    
    return out

# Input: position and velocity (format [particle, time, component]) and masses
# Output: angular momentum at each time (format [time, component])
def angular_momentum(xarray, varray, masses):
    out = np.zeros([3, xarray.shape[1]])
    for i in range(xarray.shape[0]):
        out += masses[i]*np.transpose(np.cross(xarray[i], varray[i]))
    return np.transpose(out)

# =============================================================================
# Input: initial state (6n-dim. vector), masses, time step, time of integration,
# and whether we should return the error of the measurement
# Output: time array, position over time, velocity over time, and error (if
# return_error=True)
def integrate(initial_state, masses, time_step, final_time, return_error=False):
    
    particle_num = len(initial_state)//6
    n = int(final_time/time_step)
    time = np.linspace(0, final_time, n)
    
    states_over_time = np.zeros([len(initial_state), n])
    states_over_time[:, 0] = initial_state
    
    # Implementation of RK4.
    for i in range(n-1):
        k1 = derivative(time[i], states_over_time[:, i], masses)
        k2 = derivative(time[i] + time_step/2, states_over_time[:, i] + k1*(time_step/2), masses)
        k3 = derivative(time[i] + time_step/2, states_over_time[:, i] + k2*(time_step/2), masses)
        k4 = derivative(time[i] + time_step, states_over_time[:, i] + k3*time_step, masses)
        
        states_over_time[:, i+1] = states_over_time[:, i] + (k1 + k2*2 + k3*2 + k4)*(time_step/6)
    
    # Format for x: x[particle number, position at certain time, coordinate].
    x = np.zeros([particle_num, n, 3])
    v = np.zeros([particle_num, n, 3])
    for i in range(particle_num):
        x[i, :, :] = np.transpose(states_over_time[(6*i):(6*i+3), :])
        v[i, :, :] = np.transpose(states_over_time[(6*i+3):(6*i+6), :])
    
    if return_error:
        time_dbl, x_dbl, v_dbl = integrate(initial_state, masses, time_step/2, final_time, \
                                 return_error=False)
        
        xf = x[:, -1, :]
        xf_dbl = x_dbl[:, -1, :]
        error = 0
        for i in range(particle_num):
            error += np.linalg.norm((xf - xf_dbl)[i, :])**2
        
        error = 16/15*error**0.5
        
        return time, x - COM(x, masses), v - COM(v, masses), error
    
    return time, x - COM(x, masses), v - COM(v, masses)

# =============================================================================
# Input: position, masses, whether to give a distance plot, and name of the
# saved file
# Gives a 2d plot of positions in the CM frame and saves it (if name is given)
def plot_2d(xarray, masses, plot_distance=False, figure_name=None):
    
    particle_num = xarray.shape[0]
    
    if plot_distance:
        fig, (ax, ax_dist) = plt.subplots(2, 1, figsize=[6, 10], dpi=200)
        
        r = xarray[0] - xarray[1]
        
        ax_dist.set_title('Distance Between Masses')
        ax_dist.set_xlabel('x'); ax_dist.set_ylabel('y')
        ax_dist.grid()
        ax_dist.set_aspect('equal', adjustable='box')
        
        ax_dist.plot(r[:, 0], r[:, 1], color='blue', linewidth=2)
    else:
        fig, ax = plt.subplots(dpi=200)
        
    ax.set_title('Position of Masses in the CM Frame')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid()
    ax.set_aspect('equal', adjustable='box')
    
    for i in range(particle_num):
        ax.plot(xarray[i, :, 0], xarray[i, :, 1], \
                label = f'Body {i+1} (mass {masses[i]})', color=colors[i])
        ax.plot(xarray[i, -1, 0], xarray[i, -1, 1],\
                color=colors[i], marker='o')
    
    ax.legend(loc='upper right')
    fig.show()
    
    if not (figure_name is None):
        plt.savefig(f'{figure_name}.png')

# =============================================================================
# Input: position, masses, and name of the saved file
# Gives a 3d plot of positions in the CM frame and saves it (if name is given)
def plot_3d(xarray, masses, figure_name=None):
    particle_num = xarray.shape[0]
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title("Position of Masses in the CM Frame")
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    
    for i in range(particle_num):
        ax.plot3D(xarray[i, :, 0], xarray[i, :, 1], xarray[i, :, 2],\
                label = f'Body {i+1} (mass {masses[i]})', color=colors[i])
        
        ax.plot3D(xarray[i, -1, 0], xarray[i, -1, 1], xarray[i, -1, 2], \
                color=colors[i], marker='o')
    
    ax.legend()
    fig.show()
        
    if not (figure_name is None):
        plt.savefig(f'{figure_name}.png')

# =============================================================================
# Input: position, masses, time of simulation, name of figure, and number of
# frames skipped
# Gives a 2d animation of the positions over time and saves it
def animate_2d(xarray, masses, final_time, figure_name, frame_skip=1):
    
    particle_num = xarray.shape[0]
    n = xarray.shape[1]
    dt = final_time/n
    
    fig_a = plt.figure(dpi=200)
    ax_a = plt.axes()
    ax_a.set_title('Position of Masses in CM Frame')
    ax_a.set_xlabel('x')
    ax_a.set_ylabel('y')
    ax_a.grid()
    ax_a.set_aspect('equal', adjustable='box')
    
    lines = []
    for i in range(particle_num):
        lines += [ax_a.plot(xarray[i, :, 0], xarray[i, :, 1], color=colors[i],\
                            label=f'Body {i+1} (mass {masses[i]})')]
    for i in range(particle_num):
        lines += [ax_a.plot(xarray[0, 0, 0], xarray[0, 0, 1], \
                            color=colors[i], marker='o')]
    
    def func(frame):
        real_frame = frame*frame_skip
        for i in range(particle_num):
            lines[i][0].set_data(xarray[i, :(real_frame + 1), 0], \
                               xarray[i, :(real_frame + 1), 1])
            lines[i+particle_num][0].set_data(xarray[i, real_frame, 0], \
                               xarray[i, real_frame, 1])
        
        plt.legend(title = f't = {real_frame*dt:.3g}', loc='upper right')
    
    ani = FuncAnimation(fig_a, func, frames=n//frame_skip, interval=100)
    fig_a.show()
    
    ani.save(f'{figure_name}.mp4', writer='ffmpeg', fps=30)

# =============================================================================
# Input: position, masses, time of simulation, name of figure, and number of
# frames skipped
# Gives a 3d animation of the positions over time and saves it
def animate_3d(xarray, masses, final_time, figure_name, frame_skip=1):
    
    particle_num = xarray.shape[0]
    n = xarray.shape[1]
    dt = final_time/n
    
    fig_a = plt.figure(dpi=200)
    ax_a = fig_a.add_subplot(111, projection='3d')
    
    lines = []
    for i in range(particle_num):
        lines += [ax_a.plot(xarray[i, :, 0], xarray[i, :, 1], xarray[i, :, 2], \
                            color=colors[i], label=f'Body {i+1} (mass {masses[i]})')]
    for i in range(particle_num):
        lines += [ax_a.plot(xarray[0, 0, 0], xarray[0, 0, 1], xarray[0, 0, 2],\
                     color=colors[i], marker='o')]
    
    def func(frame):
        real_frame = frame_skip*frame
        for i in range(particle_num):
            lines[i][0].set_data_3d(xarray[i][:(real_frame+1), 0], \
                                    xarray[i][:(real_frame+1), 1], \
                                    xarray[i][:(real_frame+1), 2])
            lines[i+particle_num][0].set_data_3d(xarray[i][(real_frame), 0], \
                                                   xarray[i][(real_frame), 1], \
                                                   xarray[i][(real_frame), 2])
        
        plt.legend(title = f't = {real_frame*dt:.3g}')
        return lines
    
    ani = FuncAnimation(fig_a, func, frames=n, interval=100)
    fig_a.show()
    
    ani.save(f'{figure_name}.mp4', writer='ffmpeg', fps=30)

# =============================================================================
# Input: time, COM speed, angular momentum, energy, and name of figure
# Gives a 2d plot of the deviations of conserved quantities over time, and 
# saves it (if name is given)
def conserved_quantities_plot(time, vCOM, L, E, figure_name=None):
    fig, ax = plt.subplots(3, 1, figsize=[6, 14], dpi=200)
    
    ax[0].set_title('Deviation of CM Speed From Average')
    ax[1].set_title('Deviation of Angular Momentum From Average')
    ax[2].set_title('Deviation of System Energy From Average')
    
    ax[0].set_ylabel('$\Delta v$')
    ax[1].set_ylabel('$\Delta L $')
    ax[2].set_ylabel('$ \Delta E $')
    
    for i in range(3):
        ax[i].set_xlabel('Time')
        ax[i].grid()
    
    deltaV = np.linalg.norm(vCOM - np.mean(vCOM, axis=0), axis=1)
    deltaL = np.linalg.norm(L - np.mean(L, axis=0), axis=1)
    deltaE = np.abs(E - np.mean(E))
    ax[0].legend([], title=f'CM speed: {np.linalg.norm(np.mean(vCOM, axis=0)):.3g}')
    ax[1].legend([], title=f'Angular momentum: {np.linalg.norm(np.mean(L, axis=0)):.3g}')
    ax[2].legend([], title=f'System energy: {np.mean(E):.3g}')
    ax[0].plot(time, deltaV); ax[1].plot(time, deltaL); ax[2].plot(time, deltaE)
    
    fig.show()
    if not (figure_name is None):
        plt.savefig(f'{figure_name}.png')

# =============================================================================
# Main function that does what we need for each set of initial conditions.
# Input: initial state, masses, time step, integration time, and keyword
# arguments that determine what is plotted/outputted.
def main(initial_state, masses, time_step, final_time, return_error=True, \
         plot_type='plot_2d', include_dist_in_2d=False, include_conserved_plot=False, \
             animation_frame_skip=1, figure_name="n_body_problem", conserved_name=None):
    
    if return_error:
        time, x, v, error = integrate(initial_state, masses, time_step, final_time, return_error=True)
        print(f'Estimated error: {error:.4g}')
        
    else:
        time, x, v = integrate(initial_state, masses, time_step, final_time)
    
    vCOM = COM(v, masses)
    L = angular_momentum(x, v, masses)
    E = energy(x, v, masses)
    
    if plot_type == 'plot_2d':
        plot_2d(x, masses, plot_distance=include_dist_in_2d, figure_name=figure_name)
    
    elif plot_type == 'plot_3d':
        plot_3d(x, masses, figure_name=figure_name)
    
    elif plot_type == 'animate_2d':
        animate_2d(x, masses, final_time, figure_name, frame_skip=animation_frame_skip)
    
    elif plot_type == 'animate_3d':
        animate_3d(x, masses, final_time, figure_name, frame_skip=animation_frame_skip)
    
    else:
        raise NameError('Unrecognised keyword argument')
    
    if include_conserved_plot:
        conserved_quantities_plot(time, vCOM, L, E, figure_name=conserved_name)
    
    return

# =============================================================================
# =============================================================================

from scipy.constants import G, astronomical_unit as AU
M_sun = 1.989e30
t_natural = (AU**3/(G*M_sun))**0.5

# =============================================================================
# We first show that we can simulate the two-body problem. The first three 
# examples have the same initial conditions and masses, but the velocities are 
# scaled. The fourth example has the initial conditions and masses of the
# Earth-Sun system.
# =============================================================================

# =============================================================================
# Example 1: Velocity of 0.3 units. Motion consists of two ellipses.
v0 = 0.3

m_ex1 = np.array([1, 1])
x1i_ex1 = np.array([-1, 0, 0]); v1i_ex1 = np.array([0, v0, 0])
x2i_ex1 = np.array([1, 0, 0]); v2i_ex1 = np.array([0, -v0, 0])

state_ex1 = State(x1i_ex1, v1i_ex1, x2i_ex1, v2i_ex1)

main(state_ex1, m_ex1, time_step=0.01, final_time=20, return_error=True,\
     plot_type='plot_2d', include_dist_in_2d=True, include_conserved_plot=True,\
         figure_name='ex1_motion', conserved_name='ex1_error')

# =============================================================================
# Example 2: Velocity of 1/sqrt(2) units. Motion consists of two parabolas.
v0 = 1/2**0.5

m_ex2 = np.array([1, 1])
x1i_ex2 = np.array([-1, 0, 0]); v1i_ex2 = np.array([0, v0, 0])
x2i_ex2 = np.array([1, 0, 0]); v2i_ex2 = np.array([0, -v0, 0])

state_ex2 = State(x1i_ex2, v1i_ex2, x2i_ex2, v2i_ex2)

main(state_ex2, m_ex2, time_step=0.01, final_time=20, return_error=True,\
     plot_type='plot_2d', include_dist_in_2d=True, include_conserved_plot=True,\
         figure_name='ex2_motion', conserved_name='ex2_error')

# =============================================================================
# Example 3: Velocity of 0.8 units. Motion consists of two hyperbolas.
v0 = 0.8

m_ex3 = np.array([1, 1])
x1i_ex3 = np.array([0, -1, 0]); v1i_ex3 = np.array([v0, 0, 0])
x2i_ex3 = np.array([0, 1, 0]); v2i_ex3 = np.array([-v0, 0, 0])

state_ex3 = State(x1i_ex3, v1i_ex3, x2i_ex3, v2i_ex3)

main(state_ex3, m_ex3, time_step=0.01, final_time=20, return_error=True,\
     plot_type='plot_2d', include_dist_in_2d=True, include_conserved_plot=True,\
         figure_name='ex3_motion', conserved_name='ex3_error')

# =============================================================================
# Example 4: The Earth-Sun system. Body 1 is the Sun, body 2 is the Earth. To
# show that the system has a period of 1 year = 6.28 units, we create an
# animation as well as a plot.

m_ex4 = np.array([1, 3.0025e-6])
x1i_ex4 = np.array([0, 0, 0]); v1i_ex4 = np.array([0, 0, 0])
x2i_ex4 = np.array([1, 0, 0]); v2i_ex4 = np.array([0, 1, 0])

state_ex4 = State(x1i_ex4, v1i_ex4, x2i_ex4, v2i_ex4)

main(state_ex4, m_ex4, time_step=0.01, final_time=50, return_error=True,\
     plot_type='plot_2d', include_conserved_plot=True,\
         figure_name='ex4_motion', conserved_name='ex4_error')

main(state_ex4, m_ex4, time_step=0.01, final_time=20, return_error=True,\
     plot_type='animate_2d', animation_frame_skip=10, figure_name='ex4_animation')

# =============================================================================
# We now move on to the three-body problem. Our code has been written so that
# it seamlessly generalises to n bodies, so no new code needs to be written.
# =============================================================================

# =============================================================================
# Example 5: The periodic figure-eight solution to the three-body problem. 
# Again, to show that the period is correct, we generate an animation as well
# as a static plot.

p1 = 0.347111
p2 = 0.532728

m_ex5 = np.array([1, 1, 1])
x1i_ex5 = np.array([-1, 0, 0]); v1i_ex5 = np.array([p1, p2, 0])
x2i_ex5 = np.array([0, 0, 0]); v2i_ex5 = np.array([-2*p1, -2*p2, 0])
x3i_ex5 = np.array([1, 0, 0]); v3i_ex5 = np.array([p1, p2, 0])

state_ex5 = State(x1i_ex5, v1i_ex5, x2i_ex5, v2i_ex5, x3i_ex5, v3i_ex5)

main(state_ex5, m_ex5, time_step=0.01, final_time=2000, return_error=True,\
     plot_type='plot_2d', figure_name='ex5_motion')

main(state_ex5, m_ex5, time_step=0.01, final_time=20, return_error=True,\
     plot_type='animate_2d', animation_frame_skip=6, figure_name='ex5_animation')

# =============================================================================
# Example 6: Breaking the periodic figure-eight solution. We increase the time
# step to 0.1, and note that the solution somewhat breaks after 500 units of
# time, and it completely breaks after 2757 units of time.

main(state_ex5, m_ex5, time_step=0.1, final_time=500, return_error=True,\
     plot_type='plot_2d', figure_name='ex6_motion')

# This ejection time was found through trial and error.
ejection_time = 2756.9

main(state_ex5, m_ex5, time_step=0.1, final_time=ejection_time, return_error=True,\
     plot_type='plot_2d', figure_name='ex6_ejection')
    