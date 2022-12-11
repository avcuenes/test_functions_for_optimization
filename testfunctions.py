from curses import def_prog_mode
import numpy as np 
from mealpy.bio_based import SMA


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

class rastirigin:
    def __init__(self) -> None:
        self.lb = [-5.12,-5.12,-5.12,-5.12]
        self.ub = [5.12,5.12,5.12,5.12]
        
        
    def run(self,X):
        """constraints=5.12, minimum f(0,...,0)=0"""
        A = 10
        f = 0
        n = len(X)
        for i in range(0,n):
            f = f + (X[i]**2 -A*np.cos(2*np.pi*X[i]))
        f = f + A*n
        return f

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
        self.lb = [0.05,0.25,2.0]
        self.ub = [2,1.3,15.0]
    
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
        self.ub = [2,10,10,2]
        self.P = 6000
        self.L = 14
        self.E = 30*pow(10,6)
        self.G = 12*pow(12,6)
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
        return ((4.013*self.E*np.sqrt((self.G*pow(x[2],2)*pow(x[3],6))/36))/pow(self.L,2))*(1-(x[2]/(2*self.L))*(np.sqrt(self.E /(4*self.G))))
    
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
    
    
        
   