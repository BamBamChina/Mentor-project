import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

G_F = 1.166*10**(-11) #MeV**(-2)
m_e = 0.511 #MeV
theta_W = 0.491
sin_sq_w = 0.235

def g_L(theta_W):
    return 1+2*sin_sq_w
def g_R(theta_W):
    return 2*sin_sq_w

def cross_section(T, E_nu):
    a = ((g_L(theta_W))**2)*((1-(T/E_nu))**2)
    b = g_L(theta_W)*g_R(theta_W)*m_e*T/((E_nu)**2)
    cross_section = ((G_F**2)*m_e)*((g_R(theta_W)**2)+a-b)/(2*np.pi)
    
    return cross_section

t = np.linspace(0.1, 0.3, 5) #MeV
e_nu = np.linspace(0.1, 0.5, 5) #MeV


T, E_nu = np.meshgrid(t, e_nu)

def neutrino_departure_angle(T, E_nu):
    cos_nda = 2*((E_nu)**2+(m_e)**2-T*m_e+E_nu*(T-m_e))/E_nu
    return cos_nda

mask = [[0 for i in range(len(e_nu))] for j in range(len(t))]

for i in range(len(t)):
    for j in range(len(e_nu)):
        if neutrino_departure_angle(t[i], e_nu[j]) >= -1 and neutrino_departure_angle(t[i], e_nu[j]) <= 1:
            mask[i][j] = 1
            mask[i][j] = bool(mask[i][j])
        else:
            mask[i][j] = bool(mask[i][j])
print(mask)




plt.figure(figsize=(10, 5))
sns.heatmap(cross_section(T, E_nu), xticklabels=np.round(e_nu[::], 1), 
           yticklabels=np.round(t[::-1], 2), cbar_kws={'label': 'Значение поперечного сечения'})
plt.xlabel('E_nu, MeV')
plt.ylabel('T, MeV')
plt.show()
