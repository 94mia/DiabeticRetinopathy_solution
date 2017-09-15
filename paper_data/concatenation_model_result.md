# concatenate model:

training 50 epoch

## best dme model:

binary cls accuracy: 0.9478     sensitivity: 0.9324      specificity: 0.9575

[DR binary cls]: acc: 0.9488    sensitivity: 0.9383     specificity: 0.9554

[DME binary cls]: acc: 0.8041   sensitivity: 0.5060     specificity: 0.9930
dr < 2 and dme >0 count is: 2
pred dme 0 count is: 15744
label dme 0 count is: 15645

[MIX binary cls]: acc: 0.9488   sensitivity: 0.9385     specificity: 0.9553
===> DR Kappa: 0.9129
===> Confusion Matrix:
[[11437     3   336     2     6]
 [   65    12   191     1     2]
 [  433    20  4295   203    96]
 [   14     0   480   858   229]
 [    4     0   113   151   741]]



===> DME Kappa: 0.7613
===> Confusion Matrix:
[[14740   717    72   116]
 [  778  1037   299    68]
 [  166   337   471   189]
 [   60    21   140   481]]



## best dr model:

binary cls accuracy: 0.9425	sensitivity: 0.9053	 specificity: 0.9661

[DR binary cls]: acc: 0.9436	sensitivity: 0.9132	specificity: 0.9629

[DME binary cls]: acc: 0.7521	sensitivity: 0.3641	specificity: 0.9979

[MIX binary cls]: acc: 0.9436	sensitivity: 0.9132	specificity: 0.9628
====> DR Kappa: 0.9111
[[11478    19   280     2     5]
 [   64    47   158     2     0]
 [  479   155  4036   318    59]
 [   15     6   406  1056    98]
 [    4     4   118   259   624]]

====> DME Kappa: 0.7330
[[15240   244    95    66]
 [ 1241   506   384    51]
 [  302   154   555   152]
 [  103    14   174   411]]


## best concatenation binary classification model: 

binary cls accuracy: 0.9466	sensitivity: 0.9299	 specificity: 0.9571

[DR binary cls]: acc: 0.9468	sensitivity: 0.9369	specificity: 0.9531

[DME binary cls]: acc: 0.7767	sensitivity: 0.4301	specificity: 0.9963

[MIX binary cls]: acc: 0.9468	sensitivity: 0.9369	specificity: 0.9530
====> DR Kappa: 0.9083
[[11430     0   346     1     7]
 [   59     1   209     1     1]
 [  468     0  4323   169    87]
 [   12     0   549   831   189]
 [    2     0   145   164   698]]

====> DME Kappa: 0.7367
[[15023   400   104   118]
 [ 1001   701   402    78]
 [  239   177   520   227]
 [   99    11   126   466]]




