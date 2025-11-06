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
    n2 = 30
    twidth = 50
    r = (1e6)**(1/n2)
    g = 100
    aaatol = 1e-4
    A,G,Sigma,lambdaj,res = markov_app(data, tt, dim, n2, twidth, r, g, aaatol)

    ## Error analysis


