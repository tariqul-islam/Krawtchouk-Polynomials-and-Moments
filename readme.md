<h1>Krawtchouk Polynomials and Moments</h1>

This is a collection of python codes for krawtchouk polynomials and corresponding moments.

Examples:

```
#create krawtchouk polynomials
K = wkrchkpoly(101,0.5)

#calculate moments of an image, X = input image
QS,Kr1,Kr2 = wkrchkmoment_single(X,[0.5, 0.8])

#calculate moments of a batch of image,
#X is batach of image with shape N,C,H,W
QB,Kr1,Kr2 = wkrchkmoment_batch(X,p=[0.5, 0.8])

#A second way
#preferable when reconstruction is required
N,C,H,W = X.shape
p = [0.5, 0.8]
Kr1 = wkrchkpoly(W,p[0])
Kr2 = wkrchkpoly(H,p[1])
QB = wkrchkmult_batch(X,Kr1,Kr2)

#reconstruct the batch from its moments
X = wkrchkmoment_batch_reconstruction(QB, Kr1, Kr2)

#reconstruct single image from its moments
X = wkrchkmoment_single_reconstruction(QS,Kr1,Kr2)
```