import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

G_F = 1.167*10**(-11)  #MeV
m_e = 9.1*10**(-31) #кг
tetta_W = 0.491

def g_L(tetta_W):
    return 1+2*(np.sin(tetta_W))**2
def g_R(tetta_W):
    return 2*(np.sin(tetta_W))**2

def cross_section(T, E_nu):
    a = -E_nu*((g_L(tetta_W))**2)*((1-(T/E_nu))**3)/3
    b = g_L(tetta_W)*g_R(tetta_W)*m_e*T**2/(2*(E_nu)**2)
    cross_section = (G_F*m_e)*(T*(g_R(tetta_W)**2)+a-b)/(2*np.pi)
    
    return cross_section

t = np.linspace(0.1, 12, 15) #MeV
e_nu = np.linspace(0.1, 0.5, 15) #MeV

T, E_nu = np.meshgrid(t, e_nu)



#print(cross_section(T, E_nu))

plt.figure(figsize=(10, 5))
sns.heatmap(cross_section(T, E_nu), xticklabels=np.round(e_nu[::], 2), 
           yticklabels=np.round(t[::-1], 1), cbar_kws={'label': 'Значение поперечного сечения'})
plt.xlabel('E_nu, MeV')
plt.ylabel('T, MeV')
plt.show()
