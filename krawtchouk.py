"""
Reference: Image Analysis by Krawtchouk Moments
           Pew-Thian Yap, Raveendran Paramesran, Senior Member, IEEE, and Seng-Huat Ong
               
Code Written By: Mohammad Tariqul Islam (Ponir)
                 ponir.bd@hotmail.com
"""


import numpy as np

def wkrchkpoly(N, #number of set of weighted krautchouk polynomials
               p=0.5, #p deafaults to 0.5
               ):
    """
    Kr1 = wrchkpoly(N,p=0.5)
        
    This function computes set of weighted
    krawthcouk polynomials for a given value of N
        
    Input:
        N = length and number of Krawtchouk Polynomials
        p = a value between 0 and 1
            defaults to 0.5 for centralized moments
        
    Output:
        Kr = NxN numpy matrix, each row gives a weighted krawtchouk polynomial
             first row is for n=0, second row for n=1 and so on
             
    Reference: Image Analysis by Krawtchouk Moments
               Pew-Thian Yap, Raveendran Paramesran, Senior Member, IEEE, and Seng-Huat Ong
    """
    pc = 1-p
    pr = pc/p #1-p
    x = np.array(range(0,N))
    
    #declaring size of arrays.
    w = np.zeros(N)
    rho = np.zeros(N)
    K = np.zeros((N,N))
    A = np.zeros(N)
    N=N-1 #Maximum power of the binomial function
    
    #initializing starting values
    w[0]=np.power(pc,N)
    rho[0] = 1
    K[0,:]=1
    K[1,:]=1-x/(N*p)

    for i in range(0,N-1):
        w[i+1] = w[i]*(N-i)*p/((i+1)*pc)
        rho[i+1] = 1/(-1*pr*(i+1)/(-N+i))*rho[i]
        K[i+2,:] = ((N*p+(i+1)*(1-2*p)-x)*K[i+1,:]-(i+1)*(1-p)*K[i,:])/(p*(N-i-1))

    w[N]=w[N-1]*p/((N-1+1)*pc)
    rho[N]=1/(pr*N)*rho[N-1]

    Kr=K*np.outer(np.sqrt(rho),np.sqrt(w))
    
    return Kr
    
def wkrchk2dpoly(W,H,p=[0.5,0.5]):
    """
    wkrchk2dpoly(W,H,p=[0.5,0.5])
    
    computes two dimensional weighted krawtchouk polynomias
    
    Input:
        W = Width of the polynomials
        H = Height of the polynomials
        p = an array with two members.
            first one is p value related to width
            second one is related to height
            defaults to p=[0.5, 0.5]
    
    Output:
        Z = numpy array of shape (W*H, H, W)
        Kr1 = WxW numpy matrix of weighted krawtchouk polynomials
        Kr2 = HxH numpy matrix of weighted krawtchouk polynomials

    """
    Kr1 = wkrchkpoly(W,p[0]) #calculate weighted krawtchouk moments related to width
    #if width, height and p values are equal then no need to recalculate
    if ((H==W) & (p[0]==p[1])):
        Kr2=Kr1;
    else: #otherwise recalculate
        Kr2=wkrchkpoly(H,p[1])
        
    Z = Kr2.reshape(H,H,1).dot(Kr1.reshape(W,1,W)).transpose(0,2,1,3).reshape(W*H,H,W)
    return Z, Kr1, Kr2

def wkrchkmoment_single(X,p=[0.5, 0.5]):
    """
    wkrchkmoment_single(X,p=[0.5, 0.5])
    
    Computes Krawtchouk Moments of a single grey level imagae, i.e 2D Matrix
    
    Input:
        X = 2D image, numpy array, of HxW
        p = an array with two members.
            first one is p value related to width
            second one is related to height
            defaults to p=[0.5, 0.5]
    
    Output:
        Q = HxW numpy matrix
            Q[a,b] provides the krawtchouk moments of order a+b
        Kr1 = WxW numpy matrix of weighted krawtchouk polynomials
        Kr2 = HxH numpy matrix of weighted krawtchouk polynomials
            
    """
    H,W = X.shape;
    
    Kr1 = wkrchkpoly(W,p[0])
    if (H==W & p[0]==p[1]):
        Kr2=Kr1;
    else:
        Kr2=wkrchkpoly(H,p[1])
    
    Q = Kr2.dot(X.dot(Kr1.T))
    
    return Q, Kr1, Kr2
    
def wkrchkmult_single(X,Kr1,Kr2):
    """
    wkrchkmoment_single(X,p=[0.5, 0.5])
    
    Computes Krawtchouk Moments of a single grey level imagae, i.e 2D Matrix
    
    Input:
        X = 2D image, numpy array, of HxW
        Kr1 = WxW numpy matrix of weighted krawtchouk polynomials
        Kr2 = HxH numpy matrix of weighted krawtchouk polynomials
    
    Output:
        Q = HxW numpy matrix
            Q[a,b] provides the krawtchouk moments of order a+b

            
    """
    
    Q = Kr2.dot(X.dot(Kr1.T))
    
    return Q
    
def wkrchkmoment_batch(X,p=[0.5, 0.5]):
    """
    wkrchkmoment_batch(X,p=[0.5, 0.5])
    
    Computes Krawtchouk Moments of a batch of images, called batch krawtchouk moments
    
    Input:
        X = input shape (N,C,H,W)
            interpreted as:
            N = number of images
            C = number of channel in images
            H = Height of image
            W = width of image
        p = an array with two members.
            first one is p value related to width
            second one is related to height
            defaults to p=[0.5, 0.5]
    
    Output:
        Q = output shape (N,C,H,W)
            Q[a,b,c,d] provides the krawtchouk moments of order c+d
                       of (b+1)th channel of (a+1)th image
        Kr1 = WxW numpy matrix of weighted krawtchouk polynomials
        Kr2 = HxH numpy matrix of weighted krawtchouk polynomials
            
    """
    
    N,C,H,W = X.shape;
    X1 = X.reshape(N*C,H,W)
    
    Kr1 = wkrchkpoly(W,p[0])
    if ( (H==W) & (p[0]==p[1])):
        Kr2=Kr1;
    else:
        Kr2=wkrchkpoly(H,p[1])
    
    Q = Kr2.dot(X1.dot(Kr1.T)).transpose(1,0,2).reshape((N,C,H,W))
    
    return Q, Kr1, Kr2
    
def wkrchkmult_batch(X,Kr1,Kr2):
    """
    wkrchkmult_batch(X,Kr1,Kr2)
    
    Computes Krawtchouk Moments of a batch of image, called batch krawtchouk moments
    
    Input:
        X = input shape (N,C,H,W)
            interpreted as:
            N = number of images
            C = number of channel in images
            H = height of image
            W = width of image
        Kr1 = WxW numpy matrix of weighted krawtchouk polynomials
        Kr2 = HxH numpy matrix of weighted krawtchouk polynomials
    
    Output:
        Q = output shape (N,C,H,W)
            Q[a,b,c,d] provides the krawtchouk moments of order c+d
                       of (b+1)th channel of (a+1)th image
            
    """
    N,C,H,W = X.shape;
    X1 = X.reshape(N*C,H,W)
    Q = Kr2.dot(X1.dot(Kr1.T)).transpose(1,0,2).reshape((N,C,H,W))
    
    return Q

def wkrchkmoment_batch_reconstruction(Q, Kr1, Kr2):
    """
    wkrchkmoment_batch_reconstruction(Q, Kr1, Kr2)
    
    reconstructs a batch of image from it's batch krawtchouk moments
    
    Input:
        Q = input shape (N,C,H1,W1)
            interpreted as:
            N = number of images
            C = number of channel
            H1 = height of moment matrix
            W1 = width of moment matrix
        Kr1 = WxW numpy matrix of weighted krawtchouk polynomials
        Kr2 = HxH numpy matrix of weighted krawtchouk polynomials
        
    Output:
        X = reconstructed image of shape (N,C,H,W)
            interpreted as:
            N = number of images
            C = number of channel in images
            H = height of image
            W = width of image
    
    """
    
    N,C,H1,W1 = Q.shape
    W,_ = Kr1.shape
    H,_ = Kr2.shape

    Q1 = np.pad(Q,((0,0),(0,0),(0,H-H1),(0,W-W1)),'constant', constant_values=(0,0)).reshape(N*C,H,W)
    X = Kr2.T.dot(Q1.dot(Kr1)).transpose(1,0,2).reshape(N,C,H,W)
    
    return X
    
def wkrchkmoment_single_reconstruction(Q,Kr1,Kr2):
    """
    wkrchkmoment_single_reconstruction(Q,Kr1,Kr2)
    
    reconstructs a batch of image from it's batch krawtchouk moments
    
    Input:
        Q = input shape (H1,W1)
            H1 = height of moment matrix
            W1 = width of moment matrix
        Kr1 = WxW numpy matrix of weighted krawtchouk polynomials
        Kr2 = HxH numpy matrix of weighted krawtchouk polynomials
        
    Output:
        X = reconstructed image of shape (H,W)
            interpreted as:
            H = height of image
            W = width of image
    
    """
    H1,W1 = Q.shape
    W,_ = Kr1.shape
    H,_ = Kr2.shape
    
    Q1 = np.pad(Q,((0,H-H1),(0,W-W1)),'constant', constant_values=(0,0))
    X = Kr2.T.dot(Q1.dot(Kr1))
    
    return X
