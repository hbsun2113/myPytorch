# pytorch_note

1. 查看模型名称：

   ```python
   for name, param in model.named_parameters():
       if param.requires_grad:
           print('hbsun-debug', name, param.data.shape)
   ```

   

