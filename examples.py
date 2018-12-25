### EXAMPLE 1:
# USES A FORWARD EULER TIME-STEPPING AND A LENNARD-JONES POTENTIAL TO PREVENT LEAKAGE FROM BOUNDARIES
# f DEFINES THE INITIAL SETTING OF THE PARTICLES
# THE SIMULATION RUNS FOR A TIME OF 10s, WITH A SPACIAL SPACING OF 0.8m. THE TIME-STEPPING IS ADAPTIVE
# AND ENSURES STABILITY FOR THE GIVEN PROBLEM.
import sph_fe as sph

def f(x, y):
    if 0 <= y <= 2 or (0 <= x <= 3 and 0 <= y <= 5):
        return 1
    else:
        return 0

sph.sph_simulation(x_min=[0, 0], x_max=[20, 10], t_final=10, dx=0.8, func=f, path_name='./examples/',
                   ani_step=10, ani_key="Pressure", file_name="example1")



### EXAMPLE 2:
# USES A PREDICTOR CORRECTOR TIME-STEPPING AND A LENNARD-JONES POTENTIAL TO PREVENT LEAKAGE FROM BOUNDARIES
# f DEFINES THE INITIAL SETTING OF THE PARTICLES
# THE SIMULATION RUNS FOR A TIME OF 10s, WITH A SPACIAL SPACING OF 0.8m. THE TIME-STEPPING IS ADAPTIVE
# AND ENSURES STABILITY FOR THE GIVEN PROBLEM.
import sph_ie as sph

def f(x, y):
    if 0 <= y <= 2 or (0 <= x <= 3 and 0 <= y <= 5):
        return 1
    else:
        return 0

sph.sph_simulation(x_min=[0, 0], x_max=[20, 10], t_final=10, dx=0.8, func=f, path_name='./examples/',
                   ani_step=10, ani_key="Pressure", file_name="example2")


### EXAMPLE 3:
# USES A PREDICTOR CORRECTOR TIME-STEPPING AND A LENNARD-JONES POTENTIAL TO PREVENT LEAKAGE FROM BOUNDARIES.
# ADDITIONALLY INTRODUCES AN ARTIFICIAL PRESSUE
# f DEFINES THE INITIAL SETTING OF THE PARTICLES
# THE SIMULATION RUNS FOR A TIME OF 10s, WITH A SPACIAL SPACING OF 0.8m. THE TIME-STEPPING IS ADAPTIVE
# AND ENSURES STABILITY FOR THE GIVEN PROBLEM.
import sph_ap as sph

def f(x, y):
    if 0 <= y <= 2 or (0 <= x <= 3 and 0 <= y <= 5):
        return 1
    else:
        return 0

sph.sph_simulation(x_min=[0, 0], x_max=[20, 10], t_final=10, dx=0.8, func=f, path_name='./examples/',
                   ani_step=10, ani_key="Pressure", file_name="example3")



### EXAMPLE 4:
# HERE WE SHOW HOW TO USE THE ANIMATE_RESULTS FUNCTION TO PRODUCE A PERSONALISED ANIMATION OF PRE-EXISTING FILE
# OF APPOPROPRIATE FORMAT
import animate_results as shp_ani
import matplotlib.pyplot as plt

ani = shp_ani.load_and_set('./examples/example4.csv', 'Density')
ani.animate(ani_step=10)
plt.show()