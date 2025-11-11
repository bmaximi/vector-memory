from auxfiles import *

if __name__ == '__main__':
    filenames = ["Data/vxvxc.txt", "Data/vxvyc.txt", "Data/vyvxc.txt", "Data/vyvyc.txt"]
    dim = 2
    ## Reading in the data
    with open(filenames[0], "r") as file:
        data_xx = np.loadtxt(file)
    tt = data_xx[:,0]
    data = np.zeros((dim,dim,tt.size))
    data[0,0,:] = data_xx[:,1]

    with open(filenames[1], "r") as f:
        data[0,1,:] = np.loadtxt(f)[:,1]
    with open(filenames[2], "r") as f:
        data[1,0,:] = np.loadtxt(f)[:,1]
    with open(filenames[3], "r") as f:
        data[1,1,:] = np.loadtxt(f)[:,1]

    ## Parameters for approximation
    n = 15
    n2 = 2*n
    twidth = 50
    r = (1e6)**(1/n2)
    g = 100
    aaatol = 1e-4
    A,Sigma,G,lambdaj,res,phi = markov_app(data, tt, dim, n2, twidth, r, g, aaatol)

    ## Error analysis
    app = np.zeros_like(phi)
    E = np.identity(A.shape[0])[:,:dim]
    for i in range(0,tt.size):
        app[:,:,i] = E.T@scpla.expm(A*tt[i])@Sigma@E

    errvec = npla.norm(app-phi,'fro',axis=(0,1))
    
    err1 = np.max(errvec[0:twidth*n2:twidth])# error interpolation points
    err2 = np.max(errvec[0:twidth*n2])# error interpolation interval
    err3 = np.max(errvec)# error all data points

    print('Error interpolation points:', f"{err1:.3f}")
    print('Error interpolation interval', f"{err2:.3f}")
    print('Error all data points:', f"{err3:.3f}")
    print('Number of aux. variables:',A.shape[0]-dim)



