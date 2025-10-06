import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import seaborn as sns

G_F = 1.166*10**(-11) #MeV**(-2)
m_e = 0.511 #MeV
sin_sq_w = 0.235

def g_L():
    return 1+2*sin_sq_w
    
def g_R():
    return 2*sin_sq_w

def cross_section(T, E_nu):
    a = ((g_L())**2)*((1-(T/E_nu))**2)
    b = g_L()*g_R()*m_e*T/((E_nu)**2)
    cross_section = ((G_F**2)*m_e)*((g_R()**2)+a-b)/(2*np.pi)
    
    return cross_section

t = np.linspace(0.5, 2, 25) #MeV
e_nu = np.linspace(0.5, 2, 25) #MeV


T, E_nu = np.meshgrid(t, e_nu)


data = cross_section(T, E_nu)
mask = T <= 2*(E_nu)**2/(m_e+2*E_nu)


plt.figure(figsize=(10, 5)) 
sns.heatmap(ma.array(data, mask=mask), xticklabels=np.round(e_nu[::], 1), 
           yticklabels=np.round(t[::-1], 2), cbar_kws={'label': 'Значение поперечного сечения'})
plt.xlabel('E_nu, MeV')
plt.ylabel('T, MeV')
plt.show()

