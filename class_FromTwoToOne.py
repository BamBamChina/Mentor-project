import numpy as np

class FromTwoToOne(Node):
    def __init__(self, array_cross_section, t_e, e_nu):
        super().__init__()
        self.array_cross_section = np.array(array_cross_section)
        self.t_e = np.array(t_e)
        self.e_nu = np.array(e_nu)
    def transformation(self):
        for i in range(len(self.e_nu)):
            for j in range(len(self.t_e)):
                if i == 0:
                    self.array_cross_section[i][j] = 0
                else:
                    difference = self.e_nu[i]-self.e_nu[i-1]
                    self.array_cross_section[i][j] = self.array_cross_section[i][j]*difference

        self.array_cross_section_1d = self.array_cross_section.flatten()
        return self.array_cross_section_1d


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

t_e = np.linspace(0.5, 2, 2) #MeV
e_nu = np.linspace(0.5, 2, 5) #MeV


T, E_nu = np.meshgrid(t_e, e_nu)



x = FromTwoToOne(cross_section(T, E_nu), t_e, e_nu)
print(x.transformation())
