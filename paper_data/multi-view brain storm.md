# multi-view or multi-channel or /+ multi-task brain storms

## 1. 对于分类问题，单独添加level1和level4的分级路径，是否会对结果有好的影响？

会有这个想法，是因为我猜测作为level1和level4判断依据的某些特征（微血管瘤和小的血管增生）是非常小尺度的特征，这些特征会在不断的pooling过程中逐渐消失。
为了查证下这个想法，将test集的predict值和金标准值进行了分析。分别查验了如下图像：
* predict 4, ground truth 0,1,2,3
* predict 0, ground truth 1,2,3,4
* predict 1, ground truth 0

这里发现了一个问题，目测，predict值比金标准值要准，也就是说金标准不准。当然这其中不乏很多图像质量很不好的图片。