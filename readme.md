<h1>Krawtchouk Polynomials and Moments</h1>

This is a collection of python codes for krawtchouk polynomials and corresponding moments.
Can be easily used for image processing

Examples:

```
import numpy as np
from krawtchouk import *

#create krawtchouk polynomials of length 101 and
K = wkrchkpoly(101,0.5)

#calculate moments of an image, X = input image
QS,Kr1,Kr2 = wkrchkmoment_single(X,[0.5, 0.8])

#calculate moments of a batch of images,
#X is batach of images with shape N,C,H,W
#for 5 RGB 1024x768 images shape of X is (5,3,768,1024)
#for 20 greyscale 1024x768 images shape of X is (20,1,768,1024)
QB,Kr1,Kr2 = wkrchkmoment_batch(X,p=[0.5, 0.8])

#A second way
#preferable when reconstruction is required
N,C,H,W = X.shape
p = [0.5, 0.8]
Kr1 = wkrchkpoly(W,p[0])
Kr2 = wkrchkpoly(H,p[1])
QB = wkrchkmult_batch(X,Kr1,Kr2)

#cut a number of moments from QB
#N<=H, M<=W
QBS = QB[:,:,0:N,0:M]

#reconstruct the batch from its moments
X = wkrchkmoment_batch_reconstruction(QB, Kr1, Kr2)
#or
XS = wkrchkmoment_batch_reconstruction(QBS, Kr1, Kr2)

#reconstruct single image from its moments
X = wkrchkmoment_single_reconstruction(QS,Kr1,Kr2)
```

There are some more fuctions that you can look up in the code.

Reference: Image Analysis by Krawtchouk Moments
           Pew-Thian Yap, Raveendran Paramesran, Senior Member, IEEE, and Seng-Huat Ong

-Tariqul Islam Ponir
ponir.bd @ hotmail.com