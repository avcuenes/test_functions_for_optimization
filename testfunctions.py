from curses import def_prog_mode
import numpy as np 
from mealpy.bio_based import SMA


class sphere:
    def __init__(self, dimension, lb,ub) :
        self.lb = self.lower_and_upper_bouds(dimension, lb)
        self.ub = self.lower_and_upper_bouds(dimension, ub)
    
    def lower_and_upper_bouds(self, dimension, radius):
        vector = [radius for i in range(dimension)]
        return vector
    
    def run(self,X):
        n = len(X)
        for i in range(0,n):
            f = f + X[i]*X[i]
        
        return f 


class rastirigin:
    def __init__(self,dimension, lb,ub) -> None:
        self.lb = self.lower_and_upper_bouds(dimension, lb)
        self.ub = self.lower_and_upper_bouds(dimension, ub)
    
    def lower_and_upper_bouds(self, dimension, radius):
        vector = [radius for i in range(dimension)]
        return vector
        
    def run(self,X):
        """constraints=5.12, minimum f(0,...,0)=0"""
        A = 10
        f = 0
        n = len(X)
        for i in range(0,n):
            f = f + (X[i]**2 -A*np.cos(2*np.pi*X[i]))
        f = f + A*n
        return f



      
class matyas:
    def __init__(self):
        self.lb = [-10,-10]
        self.ub = [10,10]
    
    def run(self,X):
        """constraints=10, minimum f(0, 0)=0"""
        return (0.26*(X[0]**2+X[1]**2))-(0.48*X[0]*X[1])

class eggholder:
    def __init__(self) -> None:
        self.lb = [-512,512]
        self.ub = [512,512]
        
    def run(self,X):
        """constraints=512, minimum f(512, 404.2319)=-959.6407"""
        y = X[1]+47.0
        a = (-1.0)*(y)*np.sin(np.sqrt(np.absolute((X[0]/2.0) + y)))
        b = (-1.0)*X[0]*np.sin(np.sqrt(np.absolute(X[0]-y)))
        return a+b
    
class levi:
    def __init__(self) -> None:
        self.lb = [-10,-10]
        self.ub = [10,10]

    def run(self,X):
        """constraints=10,
        minimum f(1,1)=0.0
        """
        A = np.sin(3.0*np.pi*X)**2
        B = ((X[0]-1)**2)*(1+np.sin(3.0*np.pi*X[1])**2)
        C = ((X[0]-1)**2)*(1+np.sin(2.0*np.pi*X[1])**2)
        return A + B + C


class booth:
    def __init__(self) -> None:
        self.lb = [-10,-10]
        self.ub = [10,10]
    
    def run(self,X):
        """constraints=10, minimum f(1, 3)=0"""
        return ((X[0])+(2.0*X[1])-7.0)**2+((2.0*X[0])+(X[1])-5.0)**2

class schaffer:
    def __init__(self) -> None:
        self.lb = [-100,-100]
        self.ub = [100,100]
    
    
    def run(self,X):
        """constraints=100, minimum f(0,0)=0"""
        numer = np.square(np.sin(X[0]**2 - X[1]**2)) - 0.5
        denom = np.square(1.0 + (0.001*(X[0]**2 + X[1]**2)))

        return 0.5 + (numer*(1.0/denom))



class ackley:
    def __init__(self) -> None:
        self.lb = [-5,-5]
        self.ub = [5,5]
        
    def run(self,X):
        """constraints=-5, minimum f(0,0)=0"""
        return -20*np.exp(-0.2*np.sqrt(0.5*(X[0]+X[1])))- np.exp(0.5*(np.cos(2*np.pi*X[0]) + np.cos(2*np.pi*X[1]))) + np.exp(1) + 20

class beale:
    def __init__(self) -> None:
        self.lb = [-4.5,-4.5]
        self.ub = [4.5,4.5]
    
    
    def run(self,X):
        """constraints=4.5, minimum f(0,0)=0"""
        return (1.5 - X[0] + X[0]*X[1])**2 + (2.25 - X[0] + X[0]*X[1]**2)**2 +(2.625 - X[0] +X[0]* pow(X[1],3))**2

class bukin:
    def __init__(self) -> None:
        self.lb = [-15,-3]
        self.ub = [-5,3]
        
    def run(self,X):
        """constraints=−15 ≤ x ≤ −5, −3 ≤ y ≤ 3 , minimum f(-10,1)=0"""
        return 100 * np.sqrt(np.abs(X[1]-0.01*(X[0]**2))) + 0.01 *np.abs(X[0] + 10)

class three_hump_camel:
    def __init__(self) -> None:
        self.lb = [-5,-5]
        self.ub = [5,5]
        
    def run(self,X):
        """constraints=−5 ≤ x, y ≤ 5 , minimum f(0,0)=0"""
        return 2*np.power(X[0],2) - 1.05*np.power(X[0],4) + (np.power(X[0],6)/6)  + X[0]*X[1] + np.power(X[1],2)

class easom:
    def __init__(self) -> None:
        self.lb = [-100,-100]
        self.ub = [100,100]
        
    def run(self,X):
        """constraints=−100 ≤ x, y ≤ 100 , minimum f(0,0)=0"""
        return -np.cos(X[0])*np.cos(X[1])*np.exp(-((X[0]-np.pi)**2 + (X[1]-np.pi)**2))

class mccormick:
    def __init__(self) -> None:
        self.lb = [-1.5,-3]
        self.ub = [4,4] 
    def run(self,X):
        """constraints=−1.5 ≤ x ≤ 4, −3 ≤ y ≤ 4, minimum f(0,0)=0"""
        return np.sin(X[0]+X[1]) + (X[0]-X[1])**2 - 1.5*X[0] +  2.5*X[1] +1

class leon:
    def __init__(self) -> None:
        self.lb =  [-100,-100]
        self.ub = [100,100]
    def run(self,X):
        """constraints=−100 ≤ x, y ≤ 100, minimum f(0,0)=0"""
        return 100*np.power((X[1]-np.power(X[0],2)),2) - np.power((1-X[0]) ,2)

class plateau:
    def __init__(self) -> None:
        self.lb = [-100,-100]
        self.ub = [100,100]
    def run(self,X):
        """constraints=−100 ≤ x, y ≤ 100, minimum f(0,0)=0"""
        f = 0
        n = len(X)
        for i in range(0,n):    
            f = f + X[i]
            
        return 30 + f




class pressurevessel:
    def __init__(self,alfa):
        self.alfa = alfa # 100000000
        self.lb = [0.0, 0.0,10,10]
        self.ub = [99, 99, 200,200]
    
    def run(self,x):
        fx = self.vesselfunc(x)
        beta = 2
        
        coeff= (pow(self.violate(self.g1(x)),beta)+pow(self.violate(self.g2(x)),beta) + pow(self.violate(self.g3(x)),beta) + pow(self.violate(self.g4(x)),beta))*self.alfa
        fx = fx + coeff
        return fx
    
    def printconstandobjectivefunc(self,x):
        print("f==>",self.vesselfunc(x))
        print("g1==>",self.g1(x))
        print("g2==>",self.g2(x))
        print("g3==>",self.g3(x))
        print("g4==>",self.g4(x))
        
    def violate(self,value):
            return max(0,value)
        
    def g1(self,x):
        return (-x[0] + 0.0193*x[2])

    def g2(self,x):
        return (-x[1] + 0.00954*x[2])

    def g3(self,x):
        return (-np.pi * pow(x[2],2)*x[3] - (4/3)*np.pi * pow(x[2],3) + 750*1728)

    def g4(self,x):
        return (x[3] - 240)

    def vesselfunc(self,x):
        return 0.6224*x[0]*x[2]*x[3] + 1.7781*x[1]*pow(x[2],2)+3.1661*pow(x[0],2)*x[3] + 19.84*pow(x[0],2)*x[2]

class tensioncompressionspring:
    def __init__(self,alfa):
        self.alfa = alfa #10000
        self.lb = [0.05,0.25,11.0]
        self.ub = [2,1.3,12]
    
    def run(self,x):
        fx = self.spring(x)
        
        beta = 2
        coeff = (pow(self.violate(self.g1(x)),beta) +pow(self.violate(self.g2(x)),beta) +pow(self.violate(self.g3(x)),beta) +pow(self.violate(self.g4(x)),beta))*self.alfa
        fx = fx + coeff
        
        #self.printconstandobjectivefunc(x)
        return fx
    
    def printconstandobjectivefunc(self,x):
        print("f==>",self.spring(x))
        print("g1==>",self.g1(x))
        print("g2==>",self.g2(x))
        print("g3==>",self.g3(x))
        print("g4==>",self.g4(x))

    def violate(self,value):
        return max(0,value)
        
    def spring(self,x):
        return (x[2]+2)*x[1]*pow(x[0],2)
    
    def g1(self,x):
        return 1- (pow(x[1],3)*x[2])/(71785*pow(x[0],4))
    
    def g2(self,x):
        return (4*pow(x[1],2)- x[0]*x[1])/(12566*(x[1]*pow(x[0],3)-pow(x[0],4))) + 1/(5108*pow(x[0],2)) -1
    
    def g3(self,x):
        return 1 - (140.45*x[0])/(pow(x[1],2)*x[2])
    
    def g4(self,x):
        return (x[1]+x[0])/1.5 -1
    

class weldedbeam:
    def __init__(self,alfa):
        self.alfa = alfa
        self.lb = [0.1,0.1,0.1,0.1]
        self.ub = [2,3.5,10,2]
        self.P = 6000
        self.L = 14
        self.E = 30*pow(10,6)
        self.G = 12*pow(10,6)
        self.tho_max = 13600
        self.sigma_max = 30000
        self.delta_max = 0.25
    
    def run(self,x):
        fx = self.weldedbeam(x)
        beta = 2
        
        coeff = (pow(self.violate(self.g1(x)),2) + pow(self.violate(self.g2(x)),2) +pow(self.violate(self.g3(x)),2) +pow(self.violate(self.g4(x)),2) +pow(self.violate(self.g5(x)),2) +pow(self.violate(self.g6(x)),2) +pow(self.violate(self.g7(x)),2) )*self.alfa
        
        fx = fx + coeff
        return fx
    
    def violate(self,value):
        return max(0,value)
    
    def printconstandobjectivefunc(self,x):
        print("f==>",self.weldedbeam(x))
        print("g1==>",self.g1(x))
        print("g2==>",self.g2(x))
        print("g3==>",self.g3(x))
        print("g4==>",self.g4(x))
        print("g5==>",self.g5(x))
        print("g6==>",self.g6(x))
        print("g7==>",self.g7(x))
        
        
    def weldedbeam(self,x):
        return 1.10471*pow(x[0],2)*x[1] + 0.04811*x[2]*x[3]*(14+x[1])
    
    def tho(self,x):
        tho_1 = self.P / (np.sqrt(2)*x[0]*x[1])
        M = self.P *(self.L+x[1]/2)
        R = np.sqrt(pow(x[1],2)/4 +pow((x[0]+x[2])/2,2))
        J = 2*((x[0]*x[1]*np.sqrt(2))*(pow(x[1],2)/12 +pow((x[0]+x[2])/2,2)))
        tho_2 = M*R / J 
        return np.sqrt(pow(tho_1,2)+2*tho_1*tho_2 * x[1]/(2*R) + pow(tho_2,2))
    
    def delta(self,x):
        return 4*self.P*pow(self.L,3)/(self.E*pow(x[2],3)*x[3])
    
    def sigma(self,x):
        return 6*self.P*self.L/(x[3]*pow(x[2],2))
    
    def Pc(self,x):
        return ((4.013*self.E*np.sqrt((pow(x[2],2)*pow(x[3],6))/36))/pow(self.L,2))*(1-(x[2]/(2*self.L))*(np.sqrt(self.E /(4*self.G))))
    
    def g1(self,x):
        return self.tho(x) - self.tho_max
    
    def g2(self,x):
        return self.sigma(x) - self.sigma_max
    
    def g3(self,x):
        return x[0]-x[3]
    
    def g4(self,x):
        return 0.10471*pow(x[0],2) + 0.04811*x[2]*x[3]*(14+x[1]) -5
    
    def g5(self,x):
        return 0.125 -x[0]
    
    def g6(self,x):
        return self.delta(x)-self.delta_max
    
    def g7(self,x):
        return self.P -self.Pc(x)
    
class gearbox:
    def __init__(self,alfa):
        self.alfa = alfa #10000
        self.lb = [20, 10, 30, 18, 2.75]
        self.ub = [32, 30, 40, 25, 4]
        self.i = 4
        self.rho = 8
        self.n = 6
        self.sigma = 294.3
        self.y = 0.102
        self.b_2 = 0.193
        self.K_v = 0.389
        self.K_w = 0.8
        self.N_1 = 1500
        self.P = 7.5
    
    def run(self,x):
         # input parameters
        b = x[0]
        d_1 = x[1]
        d_2 = x[2]
        Z1 = x[3]
        m = x[4]
        
        D_r = m * (self.i * Z1 - 2.5)
        l_w = 2.5*m
        D_i = D_r - 2 * l_w
        b_w = 3.5*m
        d_0 = d_2 + 25
        d_p = 0.25 * (D_i - d_0)
        D_1 = m * Z1
        D_2 = self.i * m * Z1
        N_2 = self.N_1 / self.i
        Z_2 = Z1 * D_2 / D_1
        v = np.pi * D_1 * self.N_1 / 60000
        b_1 = 102 * self.P / v
        b_3 = 4.97 * pow(10, 6) * self.P / (self.N_1*2)
        F_s = np.pi * self.K_v * self.K_w * self.sigma * b * m * self.y
        F_p = 2 * self.K_v * self.K_w * D_1 * b * Z_2 / (Z1 + Z_2)

        # constraint function
        def g1(x):
            return -F_s + b_1

        def g2(x):
            return -(F_s / F_p) + self.b_2

        def g3(x):
            return -pow(d_1, 3) + b_3

        def g4(x):
            return pow(d_2, 3)

        def g5(x):
            return (((1 + self.i) * m * Z1) / 2)

        l = 1

        def violate(value):
            return 0 if value <= 0 else value

        fx = (np.pi / 4) * (self.rho / 1000) * (
                    b * pow(m, 2) * pow(Z1, 2) * (pow(self.i, 2) + 1) - (pow(D_i, 2) - pow(d_0, 2)) * (l - b_w) - (self.n * pow(d_p, 2) * b_w) - (d_1 + d_2) * b)

        fx =fx + (violate(g1(x)) + violate(g2(x)) + violate(g3(x)))*self.alfa
        return fx
    
    def gearbox(self,x):
        b = x[0]
        d_1 = x[1]
        d_2 = x[2]
        Z1 = x[3]
        m = x[4]
        
        D_r = m * (self.i * Z1 - 2.5)
        l_w = 2.5*m
        D_i = D_r - 2 * l_w
        b_w = 3.4*m
        d_0 = d_2 + 25
        d_p = 0.25 * (D_i - d_0)
        D_1 = m * Z1
        D_2 = self.i * m * Z1
        N_2 = self.N_1 / self.i
        Z_2 = Z1 * D_2 / D_1
        v = np.pi * D_1 * self.N_1 / 60000
        b_1 = 1000 * self.P / v
        b_3 = 4.97 * pow(10, 6) * self.P / (self.N_1*1000)
        F_s = np.pi * self.K_v * self.K_w * self.sigma * b * m * self.y
        F_p = 2 * self.K_v * self.K_w * D_1 * b * Z_2 / (Z1 + Z_2)
        l = 1
        return (np.pi / 4) * (self.rho / 1000) * (b * pow(m, 2) * pow(Z1, 2) * (pow(self.i, 2) + 1) - (pow(D_i, 2) - pow(d_0, 2)) * (l - b_w) - (self.n * pow(d_p, 2) * b_w) - (d_1 + d_2) * b)
    

   
    
    
        


################################################################
###################Test Functions ##############################
################################################################
################ Taken From Grey Wolf Optimizer ################
################################################################


#### Uni Modal Benchmark Functions #############################

import random
import math

class f1:
    def __init__(self, dimension, lb,ub) :
        self.lb = self.lower_and_upper_bouds(dimension, lb)
        self.ub = self.lower_and_upper_bouds(dimension, ub)
        self.dimension = dimension
        self.name = 'f1'
        
            
    def lower_and_upper_bouds(self, dimension, radius):
        vector = [radius for i in range(dimension)]
        return vector
    
    def run(self,X):
        n = self.dimension
        f = 0
        for i in range(0,n):
            f = f + X[i]*X[i]
        
        return f 
    
    def name(self):
        return "f1"



class f2_1:
    def __init__(self,dimension,lb,ub) -> None:
        self.lb = self.lower_and_upper_bouds(dimension, lb)
        self.ub = self.lower_and_upper_bouds(dimension, ub)
        self.name = "f2_1"
        self.dimension = dimension
    
    def lower_and_upper_bouds(self, dimension, radius):
        vector = [radius for i in range(dimension)]
        return vector
    
    def run(self,X):
        f = 0
        n = self.dimension
        for i in range(0,n):
            f = f + (X[i] + 0.5)**2
        
        return f

class f2:
    def __init__(self, dimension, lb,ub) :
        self.lb = self.lower_and_upper_bouds(dimension, lb)
        self.ub = self.lower_and_upper_bouds(dimension, ub)
        self.name = "f2"
        self.dimension = dimension
    
    def lower_and_upper_bouds(self, dimension, radius):
        vector = [radius for i in range(dimension)]
        return vector
    
    def run(self,X):
        n = self.dimension
        fnc1 = 0
        fnc2 = 0 
        for i in range(0,n):
            fnc1 = fnc1 + abs(X[i])
            fnc2 = fnc2 * abs(X[i])

        return fnc1 + fnc2
    
    def name(self):
        return "f2"
    

class f3:
    def __init__(self, dimension, lb,ub) :
        self.lb = self.lower_and_upper_bouds(dimension, lb)
        self.ub = self.lower_and_upper_bouds(dimension, ub)
        self.name = "f3"
        self.dimension = dimension
    
    def lower_and_upper_bouds(self, dimension, radius):
        vector = [radius for i in range(dimension)]
        return vector
    
    def run(self,X):
        
        f = np.max(np.abs(X))   
          
        return f
    
    def name(self):
        return "f3"

class f4:
    def __init__(self, dimension, lb,ub) :
        self.lb = self.lower_and_upper_bouds(dimension, lb)
        self.ub = self.lower_and_upper_bouds(dimension, ub)
        self.name = "f4"
        self.dimension = dimension
    
    def lower_and_upper_bouds(self, dimension, radius):
        vector = [radius for i in range(dimension)]
        return vector
    
    def run(self,X):
        n = self.dimension
        f = 0
        for i in range(0,n-1):
            f = f + (100*pow((X[i+1]- pow(X[i],2)),2) + pow((X[i]-1),2))
                      
        return f
    def name(self):
        return "f4"



class f5:
    def __init__(self, dimension, lb,ub) :
        self.lb = self.lower_and_upper_bouds(dimension, lb)
        self.ub = self.lower_and_upper_bouds(dimension, ub)
        self.name = "f5"
        self.dimension = dimension
    
    def lower_and_upper_bouds(self, dimension, radius):
        vector = [radius for i in range(dimension)]
        return vector
    
    def run(self,X):
        f = 0
        n = self.dimension
        for i in range(0,n):
            f = f + pow((X[i]+0.5),2)    
          
        return f
    
    def name(self):
        return "f5"


class f6:
    def __init__(self, dimension, lb,ub) :
        self.lb = self.lower_and_upper_bouds(dimension, lb)
        self.ub = self.lower_and_upper_bouds(dimension, ub)
        self.name = "f6"
        self.dimension = dimension
    
    def lower_and_upper_bouds(self, dimension, radius):
        vector = [radius for i in range(dimension)]
        return vector
    
    def run(self,X):
        f = 0
        n = self.dimension
        for i in range(0,n):
            f = f + (i*pow(X[i],4))
        f = f + random.random()
        return f  
    
    def name(self):
        return "f6"
    
    
##### Multimodal Bechmark Functions #####


class f7:
    def __init__(self, dimension, lb,ub) :
        self.lb = self.lower_and_upper_bouds(dimension, lb)
        self.ub = self.lower_and_upper_bouds(dimension, ub)
        self.name = "f7"
        self.dimension = dimension
    
    def lower_and_upper_bouds(self, dimension, radius):
        vector = [radius for i in range(dimension)]
        return vector
    
    def run(self,X):
        n = self.dimension
        f = 0
        for i in range(0,n):
            f = f + (-1*X[i]*np.sin(np.sqrt(abs(X[i]))))
          
        return f
    
    def name(self):
        return "f7"
    


class f8:
    def __init__(self, dimension, lb,ub) :
        self.lb = self.lower_and_upper_bouds(dimension, lb)
        self.ub = self.lower_and_upper_bouds(dimension, ub)
        self.name = "f8"
        self.dimension = dimension
    
    def lower_and_upper_bouds(self, dimension, radius):
        vector = [radius for i in range(dimension)]
        return vector
    
    def run(self,X):
        n = self.dimension
        f =0
        for i in range(0,n):
            f = f + (pow(X[i],2)- 10 * (np.cos(2*np.pi*X[i])) + 10)
          
        return f
    
    def name(self):
        return "f8"
    
class f9:
    def __init__(self, dimension, lb,ub) :
        self.lb = self.lower_and_upper_bouds(dimension, lb)
        self.ub = self.lower_and_upper_bouds(dimension, ub)
        self.name = "f9"
        self.dimension = dimension
    
    def lower_and_upper_bouds(self, dimension, radius):
        vector = [radius for i in range(dimension)]
        return vector
    
    def run(self,X):
        n = self.dimension
        sumFunc =0 
        conFunc = 0
        for i in range(0,n):
            sumFunc =  sumFunc + pow(X[i],2) 
            conFunc = conFunc + np.cos(2*np.pi*X[i]) 
            
        f = -20 *  np.exp(-0.2*np.sqrt(sumFunc/n)) - np.exp(conFunc+n) + 20 +math.e 
          
        return f
    
    def name(self):
        return "f9"


class f10:
    def __init__(self, dimension, lb,ub) :
        self.lb = self.lower_and_upper_bouds(dimension, lb)
        self.ub = self.lower_and_upper_bouds(dimension, ub)
        self.name = "f10"
        self.dimension = dimension
    
    def lower_and_upper_bouds(self, dimension, radius):
        vector = [radius for i in range(dimension)]
        return vector
    
    def run(self,X):
        n = self.dimension
        func1 = 0
        func2 = 0
        for i in range(0,n):
            func1 = func1 + X[i] * X[i] 
            func2 = func2 * np.cos(X[i]/np.sqrt(i+1))
        
        f = func1/4000 - func2 
        return f
    
    def name(self):
        return "f10"

class f11:
    def __init__(self, dimension, lb,ub) :
        self.lb = self.lower_and_upper_bouds(dimension, lb)
        self.ub = self.lower_and_upper_bouds(dimension, ub)
        self.name = "f11"
        self.dimension = dimension
    
    def lower_and_upper_bouds(self, dimension, radius):
        vector = [radius for i in range(dimension)]
        return vector
    
    def run(self,X):
        n = self.dimension
        sumFunc = 0
        yfunction = 0
        for i in range(0,n):
            sumFunc =  sumFunc + self.ufunc(X,10,100,4,i)

        for i in range(0,n-1):
            yfunction = yfunction + pow((self.yfunc(X,i)-1),2)*(1+10*pow(np.sin(np.pi * self.yfunc(X,i+1)),2))
        
        lastfunction = np.pi/n *(10*np.sin(np.pi *self.yfunc(X,0))+ yfunction + pow((self.yfunc(X,n-1)-1),2)) 
          
        return lastfunction + sumFunc
    
    def name(self):
        return "f11"
    
    def ufunc(self,X,a,k,m,i):
        n = self.dimension
        
        if X[i] > a:
            return pow(k*(X[i]-a),m)
        elif -a < X[i] < a: 
            return 0
        else:
            return pow(k*(-1*X[i] -a ),m)
    
    def yfunc(self,X,i):
        func = 1 + (X[i]+1)/4
        return func



class f12:
    def __init__(self, dimension, lb,ub) :
        self.lb = self.lower_and_upper_bouds(dimension, lb)
        self.ub = self.lower_and_upper_bouds(dimension, ub)
        self.name = 'f12'
        self.dimension = dimension
    
    def lower_and_upper_bouds(self, dimension, radius):
        vector = [radius for i in range(dimension)]
        return vector
    
    def run(self,X):
        n = self.dimension
        function = 0 
        ufunction = 0
        for i in range(0,n):
            function = function + pow(X[i]-1,2)*(1+pow(np.sin(3*np.pi*X[i]+1),2))
            ufunction = ufunction + self.ufunc(X,5,100,4,i)
            
        y = pow(X[n-1]-1,2) * (1 + pow(np.sin(2*np.pi*X[n-1]),2))
        
        f = 0.1*(pow(np.sin(3*np.pi*X[0]),2) + function + y)   + ufunction     
        return f
    
    def ufunc(self,X,a,k,m,i):
        n = self.dimension
        
        if X[i] > a:
            return pow(k*(X[i]-a),m)
        elif -a < X[i] < a: 
            return 0
        else:
            return pow(k*(-1*X[i] -a ),m)
    
    def name(self):
        return "f12"
 
 
class f13:
    def __init__(self, dimension, lb,ub) :
        self.lb = self.lower_and_upper_bouds(dimension, lb)
        self.ub = self.lower_and_upper_bouds(dimension, ub)
        self.name = "f13"
        self.dimension = dimension
    
    def lower_and_upper_bouds(self, dimension, radius):
        vector = [radius for i in range(dimension)]
        return vector
    
    def run(self,X):
        n = self.dimension
        func1 = 0 
        func2 = 0
        func3 = 0
        for i in range(0,n):
            func1 = func1 + pow(np.sin(X[i]),2) 
            func2 = func2 + pow(X[i],2)
            func3 = func3 + pow(np.sin(np.sqrt(abs(X[i]))),2)
        f = (func1 - np.exp(-func2))*np.exp(-func3) 
        return f
    
    def name(self):
        return "f13"




### fixed dimension functions ###

class f14:
    def __init__(self,dimension,lb,ub):
        self.lb = self.lower_and_upper_bouds(dimension, lb)
        self.ub = self.lower_and_upper_bouds(dimension, ub)
        self.name = "f14"
        self.dimension = dimension
    
    def lower_and_upper_bouds(self, dimension, radius):
        vector = [radius for i in range(dimension)]
        return vector
    
    def run(self,X):
        """constraints=10, minimum f(1, 3)=0"""
        return ((X[0])+(2.0*X[1])-7.0)**2+((2.0*X[0])+(X[1])-5.0)**2



class f15:
    def __init__(self,dimension,lb,ub):
        self.lb = [-15,-3]
        self.ub = [-3,3]
        self.name = "f15"
        self.dimension = dimension
    
    def lower_and_upper_bouds(self, dimension, radius):
        vector = [radius for i in range(dimension)]
        return vector
        
    def run(self,X):
        """constraints=−15 ≤ x ≤ −5, −3 ≤ y ≤ 3 , minimum f(-10,1)=0"""
        return 100 * np.sqrt(np.abs(X[1]-0.01*(X[0]**2))) + 0.01 *np.abs(X[0] + 10)

class f16:
    def __init__(self,dimension,lb,ub):
        self.lb = self.lower_and_upper_bouds(dimension, lb)
        self.ub = self.lower_and_upper_bouds(dimension, ub)
        self.name = "f16"
        self.dimension = dimension
    
    def lower_and_upper_bouds(self, dimension, radius):
        vector = [radius for i in range(dimension)]
        return vector
        
    def run(self,X):
        """constraints=10, minimum f(0, 0)=0"""
        return (0.26*(X[0]**2+X[1]**2))-(0.48*X[0]*X[1])



class f17:
    def __init__(self,dimension,lb,ub):
        self.lb = self.lower_and_upper_bouds(dimension, lb)
        self.ub = self.lower_and_upper_bouds(dimension, ub)
        self.name = "f17"
        self.dimension = dimension
    
    def lower_and_upper_bouds(self, dimension, radius):
        vector = [radius for i in range(dimension)]
        return vector
        
    def run(self,X):
        """constraints=−5 ≤ x, y ≤ 5 , minimum f(0,0)=0"""
        return 2*np.power(X[0],2) - 1.05*np.power(X[0],4) + (np.power(X[0],6)/6)  + X[0]*X[1] + np.power(X[1],2)
    


class f18:
    def __init__(self,dimension,lb,ub):
        self.lb = self.lower_and_upper_bouds(dimension, lb)
        self.ub = self.lower_and_upper_bouds(dimension, ub)
        self.name = "f18"
        self.dimension = dimension
    
    def lower_and_upper_bouds(self, dimension, radius):
        vector = [radius for i in range(dimension)]
        return vector
        
    def run(self,X):
        """constraints=−100 ≤ x, y ≤ 100 , minimum f(0,0)=0"""
        return -np.cos(X[0])*np.cos(X[1])*np.exp(-((X[0]-np.pi)**2 + (X[1]-np.pi)**2))


class f19:
    def __init__(self,dimension,lb,ub):
        self.lb = self.lower_and_upper_bouds(dimension, lb)
        self.ub = self.lower_and_upper_bouds(dimension, ub)
        self.name = "f19"
        self.dimension = dimension
    
    def lower_and_upper_bouds(self, dimension, radius):
        vector = [radius for i in range(dimension)]
        return vector
        
    def run(self,X):
        """constraints=512, minimum f(512, 404.2319)=-959.6407"""
        y = X[1]+47.0
        a = (-1.0)*(y)*np.sin(np.sqrt(np.absolute((X[0]/2.0) + y)))
        b = (-1.0)*X[0]*np.sin(np.sqrt(np.absolute(X[0]-y)))
        return a+b

 


class f20:
    def __init__(self,dimension,lb,ub):
        self.lb = [-1.5,-3]
        self.ub = [4,4]
        self.name = "f20"
        self.dimension = dimension
    
    def lower_and_upper_bouds(self, dimension, radius):
        vector = [radius for i in range(dimension)]
        return vector
    
    def run(self,X):
        """constraints=−1.5 ≤ x ≤ 4, −3 ≤ y ≤ 4, minimum f(0,0)=0"""
        return np.sin(X[0]+X[1]) + (X[0]-X[1])**2 - 1.5*X[0] +  2.5*X[1] +1
    

class f21:
    def __init__(self,dimension,lb,ub):
        self.lb = self.lower_and_upper_bouds(dimension, lb)
        self.ub = self.lower_and_upper_bouds(dimension, ub)
        self.name = "f20"
        self.dimension = dimension
    
    def lower_and_upper_bouds(self, dimension, radius):
        vector = [radius for i in range(dimension)]
        return vector
    
    
    def run(self,X):
        """constraints=100, minimum f(0,0)=0"""
        numer = np.square(np.sin(X[0]**2 - X[1]**2)) - 0.5
        denom = np.square(1.0 + (0.001*(X[0]**2 + X[1]**2)))

        return 0.5 + (numer*(1.0/denom))
