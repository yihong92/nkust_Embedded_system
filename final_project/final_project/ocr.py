import pandas as pd
import paddle
from paddleocr import PaddleOCR
paddle.set_device('gpu:0')
ocr = PaddleOCR(use_angle_cls=True, cls_model_dir='path_to_cls_model')

ocr = PaddleOCR(use_gpu=True, lang='ch')
result = ocr.ocr('test.jpg', cls=True)
#225
colums = [f'Column_{i}' for i in range(5)]
df = pd.DataFrame(columns= colums,data=result)
df.to_csv('test.csv',encoding='utf-8',index=False)
print(type(result))