import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.constants import G, astronomical_unit as AU

M_sun = 1.989e30

# In our system of units, G = AU = M_sun = 1. As such, the unit of time is
# predetermined. Its value in seconds is t_natural, which is equivalent to
# about 58.1 days, or ~ 2 months.
t_natural = (AU**3/(G*M_sun))**0.5

colors = ['red', 'blue', 'green']

# =============================================================================
# The following class represents an abstraction of states in a dynamical
# system. The member data of each instance consists only of the state vector
# (represented by StateVec). Member functions retrieve the position and
# velocity of each particle in the system (the convention is
# (x1, v1, x2, v2, ...)). Addition and scalar multiplication have been
# overloaded using NumPy's definitions.
class State:
    
    # Constructor function takes an unspecified number of arguments.
    def __init__(self, *args):
        # If there is only one argument, it is assumed that this argument is
        # the state vector itself.
        if len(args) == 1:
            self.StateVec = args[0]
        # Otherwise, the arguments are assumed to be three-dimensional
        # positions and velocities. The state vector is formed by concatenating
        # all of the arguments.
        else:
            for v in args:
                if len(v) != 3:
                    raise TypeError("Inputs must be three-dimensional")
            # Concatenation.
            out = args[0]
            for v in args[1:]:
                out = np.concatenate((out, v))
                
            self.StateVec = out
    
    # Overloaded addition on states, which simply adds their state vectors.
    def __add__(self, other):
        return State(self.StateVec + other.StateVec)
    
    # Overloaded scalar multiplication on states, which simply scales their
    # state vectors.
    def __mul__(self, other):
        return State(self.StateVec * other)
    
    # Member functions that find the position and momenta of each particle in
    # the system.
    def X(self, particlenum):
        return self.StateVec[(particlenum*6) : (particlenum*6 + 3)]
    
    def V(self, particlenum):
        return self.StateVec[(particlenum*6 + 3) : (particlenum*6 + 6)]
    
    def number_of_particles(self):
        return len(self.StateVec)//6

# =============================================================================
# This function takes in the time and state, and outputs the derivative of the
# state vector for that time and state (as a State object).
def derivative(t, state, masses):
    
    N = state.number_of_particles()
    
    # An array of distances between the particles, according to the StateVec
    # in question.
    dist = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            dist[i, j] = np.linalg.norm(state.X(i) - state.X(j))
    
    # Dstate represents the derivative of the state vector. It is a list of
    # derivatives of position and velocity (Dstate = [Dx1, Dv1, Dx2, Dv2,...]),
    # so it has 2N elements. Even elements are position derivatives, and odd
    # elements are velocity derivatives.
    Dstate = []
    for i in range(N):         # For each particle i:
        
        Dstate += [state.V(i)] # The derivative of the position is the velocity
        
        Dvi = np.zeros(3)      # and the derivative of the velocity (Dvi)
        for j in range(N):     # is the sum of forces on the particle.
            if j != i:
                Dvi += masses[j]/(dist[i,j]**3)*(state.X(j) - state.X(i))
        
        Dstate += [Dvi]
    return State(*tuple(Dstate))

# =============================================================================
# This function takes an initial State object and masses, and finds the
# evolution of the system over time. It takes a time step and a final time to
# integrate the systems using the RK4 method, which has global truncation error
# of O(h^4). The various keyword arguments specify what the output is (is the
# system 2d or 3d, is it animated, are conserved values returned, etc.).
def integration(initial_state, masses, time_step, final_time, \
                plot_2d=False, plot_distance=False, animate_2d=False, \
                    plot_3d=False, animate_3d=False, frame_skip=1, \
                        return_values=False, return_error=False, \
                        save_figure=False, figure_name="n_body_problem"):
    
    particle_num = initial_state.number_of_particles()
    n = int(final_time/time_step)
    time = np.linspace(0, final_time, n)
    
    # states_over_time is a list of State objects, representing the evolution
    # of the state vector. x is a list of lists of position vectors, meaning 
    # x[i] is a list of the positions of particle i over time. Likewise for v.
    states_over_time = [initial_state]
    x = [[initial_state.X(i)] for i in range(particle_num)]
    v = [[initial_state.V(i)] for i in range(particle_num)]
    
    # Implementation of RK4.
    for i in range(n-1):
        k1 = derivative(time[i], \
                        states_over_time[i], masses)
        k2 = derivative(time[i] + time_step/2, \
                        states_over_time[i] + k1*(time_step/2), masses)
        k3 = derivative(time[i] + time_step/2, \
                        states_over_time[i] + k2*(time_step/2), masses)
        k4 = derivative(time[i] + time_step, \
                        states_over_time[i] + k3*time_step, masses)
        
        states_over_time += [states_over_time[i] \
                             + (k1+k2*2+k3*2+k4)*(time_step/6)]
        for j in range(particle_num):
            x[j] += [states_over_time[i+1].X(j)]
            v[j] += [states_over_time[i+1].V(j)]
    
    # This section implements the double-step method outlined in the report. It
    # simulates the system with h = time_step/2, and then calculates the
    # difference between the final output here and the final output above.
    if return_error:
        # Finding the final elements of the initial computation.
        xf = np.zeros([particle_num, 3])
        vf = np.zeros([particle_num, 3])
        for i in range(particle_num):
            xf[i, :] = x[i][-1]
            vf[i, :] = v[i][-1]
        
        time_dbl = np.linspace(0, final_time, 2*n)
        
        states_over_time_dbl = [initial_state]
        x_dbl = [[initial_state.X(i)] for i in range(particle_num)]
        v_dbl = [[initial_state.V(i)] for i in range(particle_num)]
        time_step_dbl = time_step/2
        for i in range(2*n-1):
            k1 = derivative(time_dbl[i], \
                            states_over_time_dbl[i], masses)
            k2 = derivative(time_dbl[i] + time_step_dbl/2, \
                            states_over_time_dbl[i] + k1*(time_step_dbl/2), masses)
            k3 = derivative(time_dbl[i] + time_step_dbl/2, \
                            states_over_time_dbl[i] + k2*(time_step_dbl/2), masses)
            k4 = derivative(time_dbl[i] + time_step_dbl, \
                            states_over_time_dbl[i] + k3*time_step_dbl, masses)
            
            states_over_time_dbl += [states_over_time_dbl[i] \
                                 + (k1+k2*2+k3*2+k4)*(time_step_dbl/6)]
            for j in range(particle_num):
                x_dbl[j] += [states_over_time_dbl[i+1].X(j)]
                v_dbl[j] += [states_over_time_dbl[i+1].V(j)]
        
        # Finding the final elements of the step-doubled computation.
        xf_dbl = np.zeros([particle_num, 3])
        vf_dbl = np.zeros([particle_num, 3])
        for i in range(particle_num):
            xf_dbl[i, :] = x_dbl[i][-1]
            vf_dbl[i, :] = v_dbl[i][-1]
        
        error = 0
        for i in range(particle_num):
            error += np.linalg.norm((xf - xf_dbl)[i, :])**2
        
        error **= 0.5
        error *= 16/15
        
    # Turning each list of positions into an array of positions.
    for i in range(particle_num):
        x[i] = np.array(x[i]); v[i] = np.array(v[i])
    
    # Format for x: x[particle number, position at certain time, coordinate].
    x = np.array(x); v = np.array(v)
    
    
    # The centre of mass, as a function of time.
    xCOM = np.cumsum(np.transpose(np.transpose(x)*masses),axis=0)[-1]
    xCOM /= sum(masses)
    vCOM = np.cumsum(np.transpose(np.transpose(v)*masses),axis=0)[-1]
    vCOM /= sum(masses)
    
    # The total energy, as a function of time.
    system_energy = np.zeros(n)
    for i in range(particle_num):
        system_energy += 0.5*masses[i]*np.linalg.norm(v[i], axis=1)**2
        for j in range(i):
            system_energy -= masses[i]*masses[j]/np.linalg.norm(x[i]-x[j], axis=1)
    
    # The total angular momentum, as a function of time.
    system_L = np.zeros([3, n])
    for i in range(particle_num):
        system_L += masses[i]*np.transpose(np.cross(x[i], v[i]))
    
    # -------------------------------------------------------------------------
    # The rest of the function depends on the various keyword arguments. It is
    # where something is outputted.
    
    # This plot is carried out if a two-dimensional distance plot is desired.
    if plot_distance:
        fig, (ax, ax_r) = plt.subplots(2, 1, figsize=[6, 10], dpi=200)
        
        r = x[0] - x[1]
        
        ax_r.set_title('Distance Between Masses')
        ax_r.set_xlabel('x')
        ax_r.set_ylabel('y')
        ax_r.grid()
        ax_r.set_aspect('equal', adjustable='box')
        ax_r.plot(r[:, 0], r[:, 1], color='blue', linewidth=2)
    
    # This plot is carried out if a static two-dimensional plot is desired.
    elif plot_2d:
        fig, ax = plt.subplots(dpi=200)
    
    if plot_2d or plot_distance:
        ax.set_title('Position of Masses in the CM Frame')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid()
        ax.set_aspect('equal', adjustable='box')
        
        for i in range(particle_num):
            ax.plot(x[i][:, 0] - xCOM[:, 0], x[i][:, 1] - xCOM[:, 1], \
                    label = f'Body {i+1} (mass {masses[i]})', color=colors[i])
            ax.plot(x[i][-1, 0] - xCOM[-1, 0], x[i][-1, 1] - xCOM[-1, 1],\
                    color=colors[i], marker='o')
        
        ax.legend(loc='upper right')
        fig.show()
        
        if save_figure == True:
            plt.savefig(f'{figure_name}.png')
    
    # This plot is carried out if a static three-dimensional plot is desired.
    if plot_3d:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_title("Position of Masses in the CM Frame")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        for i in range(particle_num):
            ax.plot3D(x[i][:, 0] - xCOM[:, 0], \
                      x[i][:, 1] - xCOM[:, 1], \
                      x[i][:, 2] - xCOM[:, 2],\
                    label = f'Body {i+1} (mass {masses[i]})', color=colors[i])
            
            ax.plot3D(x[i][-1, 0] - xCOM[-1, 0], \
                      x[i][-1, 1] - xCOM[-1, 1],\
                      x[i][-1, 2] - xCOM[-1, 2], \
                    color=colors[i], marker='o')
        
        ax.legend()
        fig.show()
        
        if save_figure == True:
            plt.savefig(f'{figure_name}.png')
        
    # This plot is carried out if a two-dimensional animation is desired.
    if animate_2d:
        x = x - xCOM
        
        fig_a = plt.figure(dpi=200)
        ax_a = plt.axes()
        ax_a.set_title('Position of Masses in CM Frame')
        ax_a.set_xlabel('x')
        ax_a.set_ylabel('y')
        ax_a.grid()
        ax_a.set_aspect('equal', adjustable='box')
        
        lines = [ax_a.plot(x[i, :, 0], x[i, :, 1], color=colors[i], label=f'Body {i+1} (mass {masses[i]})') \
                 for i in range(particle_num)] \
            + [ax_a.plot(x[0, 0, 0], x[0, 0, 1], color=colors[i], marker='o') \
               for i in range(particle_num)]
        
        def func(frame):
            for i in range(particle_num):
                lines[i][0].set_data(x[i, :(frame_skip*frame+1), 0], \
                                   x[i, :(frame_skip*frame+1), 1])
                lines[i + particle_num][0].set_data(x[i, (frame_skip*frame), 0], \
                                   x[i, (frame_skip*frame), 1])
            
            plt.legend(title = f't = {time[frame_skip*frame]:.3g}',\
                       loc='upper right')
        
        ani = FuncAnimation(fig_a, func, frames=n//frame_skip, interval=100)
        fig_a.show()
        
        ani.save(f'{figure_name}.mp4', writer='ffmpeg', fps=30)
        
    if animate_3d:
        x = x - xCOM
        fig_a = plt.figure(dpi=200)
        ax_a = fig_a.add_subplot(111, projection='3d')
        
        lines = [ax_a.plot(x[i][:, 0], x[i][:, 1], x[i][:, 2], \
                           color=colors[i], \
                           label=f'Body {i+1} (mass {masses[i]})') \
                 for i in range(particle_num)] \
            + [ax_a.plot(x[0][0, 0], x[0][0, 1], x[0][0, 2],\
                         color=colors[i], marker='o') \
               for i in range(particle_num)]
        
        time_text = ax_a.text(x=0, y=0, s='t = 0')
        
        def func(frame):
            for i in range(particle_num):
                lines[i][0].set_data_3d(x[i][:(frame_skip*frame+1), 0], \
                                        x[i][:(frame_skip*frame+1), 1], \
                                        x[i][:(frame_skip*frame+1), 2])
                lines[i + particle_num][0].set_data_3d(x[i][(frame_skip*frame), 0], \
                                                       x[i][(frame_skip*frame), 1], \
                                                       x[i][(frame_skip*frame), 2])
                
                time_text.set_text(s=f't = {time[frame_skip*frame]}')
            plt.legend()
            return lines
        
        ani = FuncAnimation(fig_a, func, frames=n, interval=100)
        fig_a.show()
        
        ani.save(f'{figure_name}.mp4', writer='ffmpeg', fps=30)
    
    if return_values and return_error:
        return time, (vCOM, np.transpose(system_L), system_energy), error
    elif return_error:
        return error
    elif return_values:
        return  time, (vCOM, np.transpose(system_L), system_energy)
    

# =============================================================================
# Finally, the following function takes in a time array and an array with vCOM,
# L and E in it, and plots their deviation from the mean as a function of time.
# The purpose of these plots is to show that each quantity is conserved.
def plot_returned_values(time, A, save_figure=True, figure_name='bruh'):
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
    
    deltaV = np.linalg.norm(A[0] - np.mean(A[0], axis=0), axis=1)
    deltaL = np.linalg.norm(A[1] - np.mean(A[1], axis=0), axis=1)
    deltaE = np.abs(A[2] - np.mean(A[2]))
    ax[0].legend(title=f'CM speed: {np.linalg.norm(np.mean(A[0], axis=0)):.3g}')
    ax[1].legend(title=f'Angular momentum: {np.linalg.norm(np.mean(A[1], axis=0)):.3g}')
    ax[2].legend(title=f'System energy: {np.mean(A[2]):.3g}')
    ax[0].plot(time, deltaV)
    ax[1].plot(time, deltaL)
    ax[2].plot(time, deltaE)
    
    fig.show()
    if save_figure==True:
        plt.savefig(f'{figure_name}.png')

#%%
# =============================================================================
# We first show that we can simulate the two-body problem. This is done by 
# using the plot_dist12 command. Our theory tells us that this should be a 
# closed ellipse, a parabola, or a hyperbola. Hence, if we can demonstrate that 
# this is the case, then it is strong evidence that our simulation works.
#
# In the first three examples, the simulation is run for 20 units of time,
# which in our system of units, is equivalent to 1838 years.
# =============================================================================

# =============================================================================
# Example 1: Two 1-unit masses, 2 units separation along the x-axis, velocity
# of 0.3 units in the +y and -y direction. The motion consists of two ellipses.
v0 = 0.3

m_ex1 = np.array([1, 1])
x1i_ex1 = np.array([-1, 0, 0]); v1i_ex1 = np.array([0, v0, 0])
x2i_ex1 = np.array([1, 0, 0]); v2i_ex1 = np.array([0, -v0, 0])

state_ex1 = State(x1i_ex1, v1i_ex1, x2i_ex1, v2i_ex1)

time1, A1, error1 = integration(state_ex1, m_ex1, time_step=0.01, final_time=20, \
            plot_2d=True, plot_distance=True, \
                save_figure=True, figure_name='ex1_motion', \
                    return_values=True, return_error=True)

plot_returned_values(time1, A1, figure_name='ex1_error')

print(f'Example 1 estimated error: {error1:.4g}')

# =============================================================================
# Example 2: Two 1-unit masses, 2 units separation along the x-axis, velocity
# of 1/sqrt(2) units in the +y and -y direction. The motion consists of two 
# parabolas.
v0 = 1/2**0.5

m_ex2 = np.array([1, 1])
x1i_ex2 = np.array([-1, 0, 0]); v1i_ex2 = np.array([0, v0, 0])
x2i_ex2 = np.array([1, 0, 0]); v2i_ex2 = np.array([0, -v0, 0])

state_ex2 = State(x1i_ex2, v1i_ex2, x2i_ex2, v2i_ex2)

time2, A2, error2 = integration(state_ex2, m_ex2, time_step=0.01, final_time=20, \
            plot_2d=True, plot_distance=True, \
                save_figure=True, figure_name='ex2_motion', \
                    return_values=True, return_error=True)

plot_returned_values(time2, A2, figure_name='ex2_error')

print(f'Example 2 estimated error: {error2:.4g}')

# =============================================================================
# Example 3: Two 1-unit masses, 2 units separation along the x-axis, velocity
# of 1 unit in the +y and -y direction. The motion consists of two hyperbolas.
v0 = 0.8

m_ex3 = np.array([1, 1])
x1i_ex3 = np.array([0, -1, 0]); v1i_ex3 = np.array([v0, 0, 0])
x2i_ex3 = np.array([0, 1, 0]); v2i_ex3 = np.array([-v0, 0, 0])

state_ex3 = State(x1i_ex3, v1i_ex3, x2i_ex3, v2i_ex3)

time3, A3, error3 = integration(state_ex3, m_ex3, time_step=0.2, final_time=2000, \
            plot_2d=True, plot_distance=True, \
                save_figure=True, figure_name='ex3_motion', \
                    return_values=True, return_error=True)

plot_returned_values(time3, A3, figure_name='ex3_error')

print(f'Example 3 estimated error: {error3:.4g}')

#%%
# =============================================================================
# Next, we show that the system adequately represents the Earth-Sun system.
# Because G = AU = M_sun = 1, the mass of the earth is 5.972e24/1.989e30
# = 3.00e-6, the displacement is exactly 1 unit, and the velocity is
# sqrt(GM_sun/R) = 1 unit.
#
# In this system, one period of the cycle is one year, which, in our system of 
# units, one year is about 6.28 units of time. We therefore integrate over 50
#  units of time, which is about eight periods.
# =============================================================================

# =============================================================================
# Example 4: The Earth-Sun system. Body 1 is the Sun, body 2 is the Earth. We
# create both a static plot and an animation for this system, as we wish to
# confirm that the period is 6.28 units of time.

m_ex4 = np.array([1, 3.0025e-6])
x1i_ex4 = np.array([0, 0, 0]); v1i_ex4 = np.array([0, 0, 0])
x2i_ex4 = np.array([1, 0, 0]); v2i_ex4 = np.array([0, 1, 0])

state_ex4 = State(x1i_ex4, v1i_ex4, x2i_ex4, v2i_ex4)

time4, A4, error4 = integration(state_ex4, m_ex4, time_step=0.01, final_time=50, \
            plot_2d=True, save_figure=True, figure_name='ex4_animation', \
                return_values=True, return_error=True)

integration(state_ex4, m_ex4, time_step=0.01, final_time=20, animate_2d=True, \
            save_figure=True, frame_skip=10, figure_name='ex4_animation')

plot_returned_values(time4, A4, figure_name='ex4_error')

print(f'Example 4 estimated error: {error4:.4g}')

#%%
# =============================================================================
# We now move on to the three-body problem. Our code has been written so that
# it seamlessly generalises to n bodies, so no new code needs to be written.
# All that needs to be done is apply the existing code to initial conditions 
# that involve three bodies.
# =============================================================================

# =============================================================================
# Example 5: The periodic figure-eight solution to the three-body problem. 
# As in the main document, the following set of initial conditions results in
# a periodic figure eight with period 6.28 units of time. We first run the
# simulation for 2000 units of time, wih a time step of 0.01. Then we create
# an animation, which goes for 20 units of time.

p1 = 0.347111
p2 = 0.532728

m_ex5 = np.array([1, 1, 1])
x1i_ex5 = np.array([-1, 0, 0]); v1i_ex5 = np.array([p1, p2, 0])
x2i_ex5 = np.array([0, 0, 0]); v2i_ex5 = np.array([-2*p1, -2*p2, 0])
x3i_ex5 = np.array([1, 0, 0]); v3i_ex5 = np.array([p1, p2, 0])

state_ex5 = State(x1i_ex5, v1i_ex5, x2i_ex5, v2i_ex5, x3i_ex5, v3i_ex5)

error5 = integration(state_ex5, m_ex5, time_step=0.01, final_time=2000, \
            plot_2d=True, save_figure=True, figure_name='ex5_motion', \
                return_error=True)

integration(state_ex5, m_ex5, time_step=0.01, final_time=20, animate_2d=True, \
            save_figure=True, frame_skip=8, figure_name='ex5_animation')

print(f'Example 5 estimated error: {error5:.4g}')

#%%
# =============================================================================
# Example 6: Breaking the periodic figure-eight solution. We increase the time
# step to 0.1, and note that the solution somewhat breaks after 500 units of
# time, and it completely breaks after 2757 units of time.

error6 = integration(state_ex5, m_ex5, time_step=0.1, final_time=500, \
            plot_2d=True, save_figure=True, figure_name='ex6_motion', \
                return_error=True)

print(f'Example 6 estimated error: {error6:.4g}')

# This ejection time was found through trial and error.
ejection_time = 2756.9

error6 = integration(state_ex5, m_ex5, time_step=0.1, final_time=ejection_time, \
            plot_2d=True, save_figure=True, figure_name='ex6_ejection', \
                return_error=True)
