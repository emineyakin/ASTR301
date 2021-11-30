import numpy as np
import matplotlib.pyplot as plt

class particle: pass

p1 = particle()

# initial conditions

e = float(input("Enter a value for e:\n"))
dt_value = float(input("Enter a value for dt:\n"))

p1.r = np.array([1.0-e, 0.0, 0.0])
p1.v = np.array([0.0, 2* np.pi * np.sqrt((1+e)/(1-e)), 0.0])
p1.a = np.zeros(3)

#simulation parameters

velocity = 0.0
distance = 0.0

energy_error = 0.0

x = []
y = []
period_values = []
energy_values = []
error_values = []


for T_p in range(1,15) :
  dt = dt_value * T_p
  x.append(p1.r[0])
  y.append(p1.r[1])
  
  r3_recip = ((p1.r[0])**2+p1.r[1]**2)**(-1.5)
  velocity = ((p1.v[0])**2 + p1.v[1]**2)**(0.5)
  distance = (p1.r[0]**2 +p1.r[1]**2)**(0.5)
  energy = - ((velocity)**2)/2 - (1/distance)

  period_values.append(T_p)
  energy_values.append(energy)

  energy_error = np.abs((energy - energy_values[T_p - 2])/ (energy_values[T_p - 2]))
  error_values.append(energy_error)

  p1.a[0] = -4*np.pi**2*p1.r[0]*r3_recip
  p1.a[1] = -4*np.pi**2*p1.r[1]*r3_recip

  p1.r += dt*p1.v
  p1.v += dt*p1.a
  print(energy_error)

  T_p += 1
  
#Plotting
  
plt.plot(np.array(x), np.array(y), "b-")
plt.show()
# Creating figure and axes
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[7, 11])

ax1.plot(np.array(period_values), np.array(energy_values), "b-")
ax1.set_title('Energy Values vs Time', fontsize=15)
ax1.set_xlabel('Time', fontsize=13)
ax1.set_ylabel('Energy values', fontsize=13)
ax1.legend()
ax1.grid()

ax2.loglog(dt * np.array(period_values), np.array(error_values), "b-")
ax2.set_title('Error Values vs Time', fontsize=15)

ax2.set_xlabel('log(Time)', fontsize=13)
ax2.set_ylabel('log(Error)', fontsize=13)

ax2.legend()
ax2.grid()

plt.tight_layout()
plt.show()
''' 
