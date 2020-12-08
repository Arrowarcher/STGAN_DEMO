## 圆秃头梦
最近一部电视剧《隐秘的角落》在网上引起了众多讨论，要说这是2020年全网热度最高的电视剧也不为过。而剧中反派Boss张东升也是网友讨论的话题之一，特别是他的秃头特点，已经成为一个梗了。
<p align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/dd9378c304ba4069906566ba95afd2264a3629bb8de94226a2ce3139215dc957" width=500 /> <br />
剧中张东升
</p>


突然很想知道自己秃头是什么样子，查了一下飞桨官网,果然它有图片生成的模型库。那么，我们如何使用paddlepaddle做出一个秃头生产器呢。
<p align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/992e8818cf9a4d16adbfe16387c5b248742e71d27d034be1ab9ad61257749e6f" width=200 /> <br />
</p>



# 图像生成
### 生成对抗网络(Generative Adversarial Network)
   **说到图像生成，就必须说到GAN，它是一种非监督学习的方式，通过让两个神经网络相互博弈的方法进行学习，该方法由lan Goodfellow等人在2014年提出。生成对抗网络由一个生成网络和一个判别网络组成，生成网络从潜在的空间(latent space)中随机采样作为输入，其输出结果需要尽量模仿训练集中的真实样本。判别网络的输入为真实样本或生成网络的输出，其目的是将生成网络的输出从真实样本中尽可能的分辨出来。而生成网络则尽可能的欺骗判别网络，两个网络相互对抗，不断调整参数。 生成对抗网络常用于生成以假乱真的图片。此外，该方法还被用于生成影片，三维物体模型等。**

### 人脸属性转换
paddle模型库里用于人脸属性转换的模型主要有三种
- StarGAN多领域属性迁移，引入辅助分类帮助单个判别器判断多个属性
	- 普通的GAN模型对每一对图像风格域都需要独立地建立一个模型，相比之下，StarGAN就是个多面手了，即单个 StarGAN 模型就可以实现多个不同风格域的转换，它允许在一个网络中同时使用不同风格域的多个数据集进行训练。这导致 StarGAN 的转化图像质量优于现有模型，并且可以灵活地转换输入图像到任何想要的风格领域。
	- StarGAN中生成网络的编码部分主要由`convolution-instance norm-ReLU`组成，解码部分主要由`transpose convolution-norm-ReLU`组成，判别网络主要由`convolution-leaky_ReLU`组成，详细网络结构可以查看`network/StarGAN_network.py`文件。生成网络的损失函数是由WGAN的损失函数，重构损失和分类损失组成，判别网络的损失函数由预测损失，分类损失和梯度惩罚损失组成。


<p align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/a7e51770815d4dcb8f19f4a350a854f360ef7f26da144d02b150a7361dd74a49" width=300 />
 <img src="https://ai-studio-static-online.cdn.bcebos.com/6b5d564e62514b54b9f3b5f45f9f7a02639e434b70f741b4ab2346dbb559d038" width=300 /> <br />
StarGAN的生成网络结构(左)和判别网络结构(右)
</p>

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200717120712881.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxODU0Mjcz,size_16,color_FFFFFF,t_70)
图 其他跨域模型与StarGAN模型的比较。
（a）为处理多个域，应该在每一对域都建立跨域模型。（b）StarGAN用单个generator学习多域之间的映射。该图表示连接多个域的拓扑图。


- AttGAN利用分类损失和重构损失来保证改变特定的属性
	- AttGAN算法是基于encoder-decoder结构的，根据所需属性对给定人脸的潜在表征进行解码，实现人脸属性的编辑。现有些方法试图建立一个独立于属性的潜在表示，来编辑属性。然而，这种对潜在表征的属性无关约束过多，限制了潜在表征的能力，可能导致信息丢失，从而导致生成过于平滑和扭曲。AttGAN没有对潜在表示施加约束，而是对生成的图像应用属性分类约束(attribute classification constraint)，以保证所需属性的正确改变，即“改变你想要的”。同时，引入重构损失来保证只改变特定的属性。还可以直接应用于属性强度控制，并且可以自然地扩展到属性样式操作。
	- AttGAN中生成网络的编码部分主要由`convolution-instance norm-ReLU`组成，解码部分由`transpose convolution-norm-ReLU`组成，判别网络主要由`convolution-leaky_ReLU`组成，详细网络结构可以查看`network/AttGAN_network.py`文件。生成网络的损失函数是由WGAN的损失函数，重构损失和分类损失组成，判别网络的损失函数由预测损失，分类损失和梯度惩罚损失组成。


<p align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/af1a6fd37a864aa7ba37e1a620cc394a261116679f48419782a702954e6fab06" width=800 /> <br />
AttGAN的网络结构
</p>

- STGAN只输入有变化的标签，引入GRU结构，更好的选择变化的属性
	- STGAN是在AttGAN的基础上做的改进。STGAN中生成网络在编码器和解码器之间加入Selective Transfer Units(STU)，同时引入属性差异向量(只输入需要改变的属性)，这样一来，网络变得更容易训练，相比于目标属性标签，属性差异标签可以提供更多有价值的信息，使属性生成精度明显提升。
	- STGAN中生成网络再编码器和解码器之间加入Selective Transfer Units\(STU\)，有选择的转换编码网络，从而更好的适配解码网络。生成网络中的编码网络主要由`convolution-instance norm-ReLU`组成，解码网络主要由`transpose convolution-norm-leaky_ReLU`组成，判别网络主要由`convolution-leaky_ReLU`组成，详细网络结构可以查看`network/STGAN_network.py`文件。生成网络的损失函数是由WGAN的损失函数，重构损失和分类损失组成，判别网络的损失函数由预测损失，分类损失和梯度惩罚损失组成。

<p align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/88d3bb1b50a749a59742e2726b31ad067a31c62d1bc8482c991d949fcf41e2e1" width=800 /> <br />
STGAN的网络结构
</p>

>STGAN差不多是AttGAN的升级版，StarGAN不支持秃头属性，所以我们使用`STGAN`。

## STGAN效果预览

<p align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/4f0745b3394a4cafbb0c228dff6f2384f27c104011444445b73da6b9e29d53b5" width="1250"/><br />
STGAN的效果图(图片属性分别为：original image, Bald, Bangs, Black Hair, Blond Hair, Brown Hair, Bushy Eyebrows, Eyeglasses, Male, Mouth Slightly Open, Mustache, No Beard, Pale Skin, Young)
</p>
发现里面第一个就是秃头

本项目采用celeba数据集，关于celeba数据集的介绍，详见https://zhuanlan.zhihu.com/p/35975956




```python
# 解压数据集
# !unzip data/data21325/imgAlignCeleba.zip -d dataset/
# !cp data/data21325/*.txt -d dataset/

```


```python
# 获取模型（我已经把需要的文件放在work里,不用再获取）
# !git clone https://gitee.com/paddlepaddle/models.git -b release/1.8
# !cp -r models/PaddleCV/gan/* ./work/
```


```python
# 训练，我已经花费近18个小时训练了一个，可以直接使用了，所以略过这一步吧
%cd ~/dataset
!python ../work/train.py --model_net STGAN \
--data_dir ../dataset \
--dataset . \
--crop_size 170 \
--image_size 128 \
--train_list ../dataset/attr_celeba.txt \
--gan_mode wgan  \
--batch_size 32 --print_freq 1 \
--num_discriminator_time 5 \
--epoch 50 \
--dis_norm instance_norm \
--output ~/output/stgan/
```


```python
# 解压训练好的模型
!unzip data/data43743/stgan.zip -d ~/
```

    Archive:  data/data43743/stgan.zip
       creating: /home/aistudio/33/
       creating: /home/aistudio/33/.ipynb_checkpoints/
      inflating: /home/aistudio/33/net_G.pdmodel  
      inflating: /home/aistudio/33/net_G.pdopt  
      inflating: /home/aistudio/33/net_G.pdparams  


## 别着急
在“秃头”之前，我们需要先准备要输入的图片，我把他放在`my_dataset`里，并且修改`dataset/test1.txt`，把图片填进去，并且根据图片的特征输入特征
![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9haS1zdHVkaW8tc3RhdGljLW9ubGluZS5jZG4uYmNlYm9zLmNvbS84YjIyODkyODIzYTc0NWYwOGQ1ODBlZTEzMzRjNTM1MTY0MWUzNjEwNzhiYjQ1ZDFiOTU4M2NmNGMwOWE0Y2Jk?x-oss-process=image/format,png)




```python
%cd ~
# 输入的参数可以看看infer_bald.py开头的解释哦，主要需要注意的是n_samples、crop_size、image_size
# crop_size、image_size最好不要修改，经过我测试会影响效果
!python ./work/infer_bald.py \
--model_net STGAN \
--init_model ./33/ \
--dataset_dir my_dataset \
--test_list dataset/test1.txt \
--use_gru True \
--output ./infer_result/stgan/ \
--n_samples 1 \
--selected_attrs "Bald,Bangs,Black_Hair,Blond_Hair,Brown_Hair,Bushy_Eyebrows,Eyeglasses,Male,Mouth_Slightly_Open,Mustache,No_Beard,Pale_Skin,Young" \
--c_dim 13 \
--crop_size 178 \
--image_size 128 \
--load_height 228 \
--load_width 228 \
--crop_height 128 \
--crop_width 128 \


```

## 秃头效果展示

<p align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/89f20e7f715b4a959aee089140a1befcb0c16b54f801435b968071c96a89875d" width=800/> <br />
挺秃然的。。
</p>
<p align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/8f16410e9c7f43be8f6a526d4cda56302899b960e2a8429b9c571927201b51b2" width=800 /> <br />
挺秃然的。。
</p>
<p align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/aa1975c1871841dc9f3b1399097c254b4636e2e8de594fb5b92d38ac8089f35b" width=800 /> <br />
秦昊
</p>

## 效果结论
* 因为stgan只用输入变化的属性，原infer会循环变化每个一个风格属性(Bald，Bangs等)，我把切换风格属性的循环，修改成了只输入秃头属性，然后循环逐渐改变变化程度，使其结果产生渐变效果
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200717120921300.png)
* 这里我发现男性中头发是短发的秃头化效果，明显比长发男性好，头发蓬松就会影响秃头效果，因为蓬松的头发遮盖了脑袋的形状，也有可能是训练集缺少这类男性图片的原因。第一张图是最自然的，看起来也最真，应该是寸头短发的因素。
* 输入图片的大小接近128x128，或者178x178，效果会比较好，原因可能是训练集的大小都是128x128
	- [https://www.paddlepaddle.org.cn/hubdetail?name=stgan_celeba&en_category=GANs](http://)
   - 官方这里也说明：
    	- 待处理图片尽量只露脸，当五官是朝向正前方且露出五官时，效果会比较好。
      - 待处理图片的尺寸接近 128 * 128 像素时，效果会比较好。


```python
## 使用paddlehub
如果觉得上面的比较繁琐，infer里的代码复杂，那么有一条直接的捷径。
paddlehub里面已经有stgan的预训练模型。
```


```python
# 安装paddlehub和stgan_celeba预训练模型
!pip install paddlehub==1.6.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
!hub install stgan_celeba
```


```python
import paddlehub as hub

stgan = hub.Module(name="stgan_celeba")

test_img_path = ["my_dataset/img_align_celeba/000003.jpg"]
# org_info是一个只有一个元素的列表 如：["Bald,Bangs"]
# org_info要尽可能详细的说明输入图片的特征情况,否则会影响输出效果：
#  必须填写性别（ "Male" 或 "Female"）可选值"Bald", "Bangs",
# "Black_Hair", #"Blond_Hair", "Brown_Hair", "Bushy_Eyebrows", 
# "Eyeglasses", #"Mouth_Slightly_Open", "Mustache", "No_Beard", "Pale_Skin", "Aged"
org_info = ["Male"]
# 指定要变化的特征：秃头
trans_attr = ["Bald"]

# set input dict
input_dict = {"image": test_img_path, "style": trans_attr, "info": org_info}

# execute predict and print the result
results = stgan.generate(data=input_dict)
print the result
results = stgan.generate(data=input_dict)
print(results)
```

## 预训练模型和我自己训练的模型对比
<p align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/89f20e7f715b4a959aee089140a1befcb0c16b54f801435b968071c96a89875d" width=800 /> <br />
我自己的
</p>

<p align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/f52e73177243470d995251fc604f596da77d7494a29c4dda8a1766a6fa7ea3bd" width=200 /> <br />
预训练模型的
</p>

* 可以看出预训练模型的秃头程度比较固定，而我把秃头程度设置的比较小，看起来可能更真一丢丢~~

### 项目体验地址
https://aistudio.baidu.com/aistudio/projectdetail/620058?shared=1

### 感谢
最后感谢飞桨平台，让我这个初学者就能做一些有趣的试验。

还有我对stylegan挺感兴趣的，希望之后可以支持到哈


AI初学经历：
《百度架构师手把手教深度学习》:
https://aistudio.baidu.com/aistudio/education/group/info/888?activityId&shared=1
《强化学习7日打卡营》:
https://aistudio.baidu.com/aistudio/education/group/info/1335?activityId&shared=1

如果您加入官方QQ群，您将遇上大批志同道合的深度学习同学。飞桨官方QQ群：1108045677。
如果您想详细了解更多飞桨的相关内容，请参阅以下文档。
官网地址：https://www.paddlepaddle.org.cn

感兴趣的同学还可以看看

PaddleGAN:https://github.com/Joejiong/PaddleGAN

欢迎Star

飞桨开源框架项目地址：
GitHub: https://github.com/PaddlePaddle/Paddle
Gitee:  https://gitee.com/paddlepaddle/Paddle

飞桨生成对抗网络项目地址：
GitHub: 
https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleCV/gan
Gitee: 
https://gitee.com/paddlepaddle/models/tree/develop/PaddleCV/gan
