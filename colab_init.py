# 从github克隆共享代码
# !rm -rf quant
# !git clone https://github.com/northcheng/quant.git   
  
!pip install pandas_datareader
!pip install https://github.com/matplotlib/mpl_finance/archive/master.zip

# 挂上Google Drive
from google.colab import drive
drive.mount('/content/drive')
!ls drive/My\ Drive/

# 自定义函数
from quant import bc_util as util
from quant import bc_colab as colab_util