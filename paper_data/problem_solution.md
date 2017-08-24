# Problem Solution

1. ```logger = cls_train(dataset_train, nn.DataParallel(model).cuda(), criterion, optimizer, epoch, opt.display)```
注意 ```nn.DataParallel(model).cuda()``` 
不要使用```model_cuda = nn.DataParallel(model).cuda()``` 
```logger = cls_train(dataset_train, model_cuda, criterion, optimizer, epoch, opt.display)```

