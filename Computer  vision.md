### Computer Vision

Computer vision, machine vision and image processing are to process image in computer for different visual tasks from diverse perspectives.
Computer vision focus on the visual representation includes images, graphics,
animation and videos rather than the sounds, speech or text.  

[Visual recognition tasks, such as image classification, localization, and detection, are the core building blocks of many of
these applications, and recent developments in Convolutional Neural Networks (CNNs) have
led to outstanding performance in these state-of-the-art visual recognition tasks and systems.
As a result, CNNs now form the crux of deep learning algorithms in computer vision.](https://ieeexplore.ieee.org/document/8295029)
[The Ancient Secret of Computer Vision](https://pjreddie.com/courses/computer-vision/) covers standard techniques in image processing like filtering, edge detection, stereo, flow, etc. (old-school vision), as well as newer, machine-learning based computer vision, which is more comprehensive.

<img title="cvcl" src="http://cvcl.mit.edu/imagesAude/triangle2.png" width="80%" />

* https://sites.google.com/visipedia.org/index
* [Computational Vision at Caltech](http://www.vision.caltech.edu/)
* [SE(3) COMPUTER VISION GROUP AT CORNELL TECH](https://vision.cornell.edu/se3/publications/)
* https://vcla.stat.ucla.edu/index.html
* [香港中文大学多媒体实验室](http://mmlab.ie.cuhk.edu.hk/index_cn.html)
* [计算机视觉与遥感实验室 Computer Vision & Remote Sensing (CVRS) Lab@WHU](http://cvrs.whu.edu.cn/)
* [Center for Research in Computer Vision](https://www.crcv.ucf.edu/)
* [Vision and Content Engineering Lab@ETZH](http://www.kalisteo.eu/en/index.htm)
* [Computer Vision and Geometry Group](https://cvg.ethz.ch/)
* https://github.com/jbhuang0604/awesome-computer-vision
* [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
* [Spring 2019 CS 543/ECE 549: Computer Vision](http://slazebni.cs.illinois.edu/spring19/)
* [Elements of Geometric Computer Vision. by Andrea Fusiello ](http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/FUSIELLO4/tutorial.html)
* [CVonline: The Evolving, Distributed, Non-Proprietary, On-Line Compendium of Computer Vision](http://homepages.inf.ed.ac.uk/rbf/CVonline/)
* [An Introduction to Projective Geometry  (for computer vision) by Stan Birchfield](http://robotics.stanford.edu/~birch/projective/)
* [Photogrammetric Computer Vision -- Statistics, Geometry, and Reconstruction](http://www.ipb.uni-bonn.de/book-pcv/)
* http://slazebni.cs.illinois.edu/spring17/lec01_cnn_architectures.pdf
* https://blog.algorithmia.com/introduction-to-computer-vision/
* https://hayo.io/computer-vision/
* http://www.robots.ox.ac.uk/~vgg/research/text/
* https://ieeexplore.ieee.org/document/8295029
* http://www.vlfeat.org/matconvnet/
* http://cvcl.mit.edu/aude.htm

<img title="hayo.io" src="https://hayo.io/under_development/wp-content/uploads/2017/01/shema1-2.jpg" width="50%" />


Image acquisition | Image processing | Image analysis|
---|---|-----
 Webcams & embedded cameras |Edge detection| 3D scene mapping
Digital compact cameras & DSLR |Segmentation| Object recognition
Consumer 3D cameras  | Classification | Object tracking
 Laser range finders |Feature detection and matching|---


- [ ] [9 Applications of Deep Learning for Computer Vision](https://machinelearningmastery.com/applications-of-deep-learning-for-computer-vision/)
- [ ] [Deep Learning for Computer Vision
Image Classification, Object Detection, and Face Recognition in Python](https://machinelearningmastery.com/deep-learning-for-computer-vision/)
- [ ] http://modelnet.cs.princeton.edu/

#### Image Classification / Recognition

Like other classification  tasks, the feature engineering is the core and kernel of preprocessing.
The predicted labels is always supposed to be attached to some specific features. As  the fingerprint can be an biological identifier of its owner, the desired features are supposed to be sufficient to identify the differences between the ones with the same label.


* https://cv-tricks.com/cnn/understand-resnet-alexnet-vgg-inception/
* https://www.jeremyjordan.me/convnet-architectures/
* http://www.tamaraberg.com/teaching/Spring_16/

**LeNet**

`LeNet` learns the parameters using error back-propagation. In another word, the optimization procedure of  CNNs are based on gradient methods. This CNN
model was successfully applied to recognize handwritten digits.

![LeNet](https://engmrk.com/wp-content/uploads/2018/09/LeNet_Original_Image.jpg)

As shown in the above figure, it consists of convolution, subsampling, full connection and Gaussian connections.
It is a typical historical architecture.

**AlexNet**

`AlexNet` was the winning entry in [ILSVRC](http://www.image-net.org/challenges/LSVRC/) 2012. It solves the problem of image classification where the input is an image of one of 1000 different classes (e.g. cats, dogs etc.) and the output is a vector of 1000 numbers.

<img src="https://vitalab.github.io/deep-learning/images/alexnet/alexnet.jpg" width="50%" />

<img title="Alex-Net" src="https://www.learnopencv.com/wp-content/uploads/2018/05/AlexNet-1.png" width="50%"/>

AlexNet consists of 5 Convolutional Layers and 3 Fully Connected Layers.
`Overlapping Max Pool` layers are similar to the Max Pool layers, except the adjacent windows over which the max is computed overlap each other.
An important feature of the AlexNet is the use of `ReLU(Rectified Linear Unit)` Nonlinearity.
* https://engmrk.com/alexnet-implementation-using-keras/
* http://gyxie.github.io/2016/09/21/%E6%B7%B1%E5%85%A5AlexNet/
* [Understanding AlexNet](https://www.learnopencv.com/understanding-alexnet/)
* [ImageNet Classification with Deep Convolutional Neural Networks](http://vision.stanford.edu/teaching/cs231b_spring1415/slides/alexnet_tugce_kyunghee.pdf)

**VGG**

[The very deep ConvNets were the basis of our ImageNet ILSVRC-2014 submission, where our team (VGG) secured the first and the second places in the localisation and classification tasks respectively.](http://www.robots.ox.ac.uk/~vgg/research/very_deep/)

<img title="VGG" src="https://srdas.github.io/DLBook/DL_images/VGGNet.jpg" width="50%"/>

* http://www.robots.ox.ac.uk/~vgg/practicals/cnn/index.html
* http://www.robots.ox.ac.uk/~vgg/research/very_deep/
* [VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION](https://arxiv.org/pdf/1409.1556.pdf)

**Inception**

<img title="Inception" src="https://srdas.github.io/DLBook/DL_images/Inception1.jpg" width="50%"/>

* [Deep Learning in the Trenches: Understanding Inception Network from Scratch](https://www.analyticsvidhya.com/blog/2018/10/understanding-inception-network-from-scratch/)
* http://www.csc.kth.se/~roelof/deepdream/bvlc_googlenet.html
* https://zh.d2l.ai/chapter_convolutional-neural-networks/googlenet.html

**ResNet**

`ResNet` is to solve the degeneration of deep neural  network when the depth of layers of deep neural network increase.
It is guessed that the nonlinearity of activation function makes it tough to learn (linear) identity transformation \(Id(x)=x\).

<img title="VGG VS. Res" src="https://neurohive.io/wp-content/uploads/2019/01/resnet-architecture.png" width="50%" />

* https://neurohive.io/en/popular-networks/resnet/
* http://teleported.in/posts/decoding-resnet-architecture/
* https://zh.d2l.ai/chapter_convolutional-neural-networks/resnet.html

**DenseNet**

![DenseNet](http://openresearch.ai/uploads/default/original/1X/a3bd62739f80a8faf6b92861bf82ace09201c7ee.png)

* [Densely Connected Convolutional Networks](http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf)
* [Dive to Deep Learning](https://zh.d2l.ai/chapter_convolutional-neural-networks/densenet.html)



#### Semantic Segmentation

* http://cvlab.postech.ac.kr/research/deconvnet/
* https://deeplearninganalytics.org/semantic-segmentation/
* http://www.cs.toronto.edu/~tingwuwang/semantic_segmentation.pdf
* http://wp.doc.ic.ac.uk/bglocker/project/semantic-imaging/
* [A 2017 Guide to Semantic Segmentation with Deep Learning](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review)
* [Semantic Image Segmentation Live Demo](http://www.robots.ox.ac.uk/~szheng/crfasrnndemo)
* [Awesome Semantic Segmentation](https://github.com/mrgloom/awesome-semantic-segmentation)

#### Object Detection

<img title ="Objection Detection" src="https://img-blog.csdn.net/20180502184736909" width="50%"/>

**RCNN**

<img src="https://img-blog.csdn.net/20170324121024882" width="50%"/>

* [RCNN, Fast RCNN, Faster RCNN 总结](http://shartoo.github.io/RCNN-series/)
* https://www.cnblogs.com/skyfsm/p/6806246.html
* [YOLO 1 到 YOLO 3](http://shartoo.github.io/yolo-v123/)

***
* https://www.jeremyjordan.me/object-detection-one-stage/
* https://github.com/amusi/awesome-object-detection
* [Object Detection 2015](https://handong1587.github.io/deep_learning/2015/10/09/object-detection.html)
* https://blog.csdn.net/v_JULY_v/article/details/80170182

#### Optical Character Recognition

<img src="https://pic1.zhimg.com/80/v2-a8fe4dc5455d692d273f06be2bb24d30_hd.jpg" width="50%" />

* https://indic-ocr.github.io/
* https://github.com/kba/awesome-ocr
* https://github.com/pannous/tensorflow-ocr
* https://zhuanlan.zhihu.com/p/65707543
* https://github.com/tesseract-ocr/tesseract
* https://github.com/Swift-AI/Swift-AI
* https://github.com/wanghaisheng/awesome-ocr
* https://github.com/chineseocr
* http://tesseract-ocr.github.io/4.0.0/index.html
* https://anyline.com/
* https://github.com/vinayakkailas/Deeplearning-OCR
* https://github.com/hs105/Deep-Learning-for-OCR
* http://cs231n.stanford.edu/reports/2017/pdfs/810.pdf
* https://handong1587.github.io/deep_learning/2015/10/09/ocr.html
* [STN-OCR: A single Neural Network for Text Detection and Text Recognition](https://arxiv.org/abs/1707.08831)

#### Object Tracking

* https://cv-tricks.com/object-tracking/quick-guide-mdnet-goturn-rolo/

#### Pose Estimation

* https://www.fritz.ai/features/pose-estimation.html
* https://github.com/xinghaochen/awesome-hand-pose-estimation
* https://github.com/cbsudux/awesome-human-pose-estimation
* https://cv-tricks.com/pose-estimation/using-deep-learning-in-opencv/

#### Image Caption

- https://cs.stanford.edu/people/karpathy/sfmltalk.pdf
- https://www.ripublication.com/ijaer18/ijaerv13n9_102.pdf
- [Deep Visual-Semantic Alignments for Generating Image Descriptions](https://github.com/karpathy/neuraltalk2)
- [Automatic Image Captioning using Deep Learning (CNN and LSTM) in PyTorch](https://www.analyticsvidhya.com/blog/2018/04/solving-an-image-captioning-task-using-deep-learning/)
- [How to Develop a Deep Learning Photo Caption Generator from Scratch](https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/)
- [Deep Learning Image Caption Generator](https://github.com/damminhtien/Deep-Learning-Image-Caption-Generator)
- [Conceptual Captions: A New Dataset and Challenge for Image Captioning
Wednesday, September 5, 2018](https://ai.googleblog.com/2018/09/conceptual-captions-new-dataset-and.html)

#### Scene Understanding

- [Towards Total Scene Understanding: Classification, Annotation and Segmentation in an Automatic Framewor](http://vision.stanford.edu/projects/totalscene/index.html)
- [6.870 Object Recognition and Scene Understanding Fall 2008](http://people.csail.mit.edu/torralba/courses/6.870/6.870.recognition.htm)
- http://vladlen.info/projects/scene-understanding/
- [Holistic Scene Understanding](https://ttic.uchicago.edu/~yaojian/HolisticSceneUnderstanding.html)
- http://www.kalisteo.eu/en/thematic_vs.htm
- https://ps.is.tue.mpg.de/research_fields/semantic-scene-understanding
- [ScanNet Indoor Scene Understanding Challenge
CVPR 2019 Workshop, Long Beach, CA](http://www.scan-net.org/cvpr2019workshop/)
- [Human-Centric Scene Understanding from Single View 360 Video](https://cvssp.org/projects/s3a/AffordRecon/)
- [HoloVis - 3D Holistic Scene Understanding](http://www.l3s.de/en/node/1133)
- [Scene Recognition and Understanding](http://sunai.uoc.edu/index.php/2016/02/07/scene-recognition-and-understanding/)
- [BlitzNet: A Real-Time Deep Network for Scene Understanding](http://thoth.inrialpes.fr/research/blitznet/)
- [L3ViSU: Lifelong Learning of Visual Scene Understanding](http://cvml.ist.ac.at/erc/)

#### Image Search

- [ ] [Modeling visual knowledge from large-scale data](http://lear.inrialpes.fr/research.php)
- [ ] [Deep Convolutional Matching](https://hal.inria.fr/hal-01148432v1)


#### Style Transfer

<img title="pyimagesearch" src="https://www.pyimagesearch.com/wp-content/uploads/2018/08/neural_style_transfer_gatys.jpg" width="60%"/>

* https://github.com/titu1994/Neural-Style-Transfer
* https://pjreddie.com/darknet/nightmare/
* https://deepdreamgenerator.com/
* https://www.pyimagesearch.com/2018/08/27/neural-style-transfer-with-opencv/
* https://reiinakano.github.io/2019/01/27/world-painters.html
* https://www.andrewszot.com/blog/machine_learning/deep_learning/style_transfer
* [Convolutional Neural Network – Exploring the Effect of Hyperparameters and Structural Settings for Neural Style Transfe](https://lanstonchu.wordpress.com/2018/09/03/convolutional-neural-network-exploring-the-effect-of-hyperparameters-and-structural-settings-for-neural-style-transfer/)


#### Computer Graphics

* https://github.com/alecjacobson/computer-graphics-csc418
* http://graphics.cs.cmu.edu/courses/15-463/
* https://github.com/ivansafrin/CS6533s
* https://github.com/ericjang/awesome-graphics

***
* http://kvfrans.com/coloring-and-shading-line-art-automatically-through-conditional-gans/

|Deep Dream|
|:--------:|
|![Deep Dream](http://grayarea.org/wp-content/uploads/2016/03/3057368-inline-i-1-inside-the-first-deep-dream-art-show.jpg)|
****

[Deep Learning for Computer Vision, Speech, and Language](https://columbia6894.github.io/)
